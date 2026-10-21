"""Verification, AHRI 550/590 rating, engineering benchmarks, sensitivity, ASHRAE Guideline 14 statistics."""
import numpy as np
from building import Building
from chiller import Plant
from simulate import simulate


def integrator_vs_analytical(R, day=3):
    b = R['bld']; dt = R['dt']; n = int(24 / dt)
    s = day * n
    u = np.column_stack([R['Tdb'][s:s + n], R['Qa'][s:s + n], R['Qm'][s:s + n], R['q_sens'][s:s + n]])
    x_rk = np.array([R['Ta'][s], R['Tm'][s]]); x_ex = x_rk.copy(); err = []
    for k in range(n):
        x_rk = b.step_rk4(x_rk, u[k], dt); x_ex = b.step_exact(x_ex, u[k], dt)
        err.append(abs(x_rk - x_ex).max())
    return float(max(err))


def energy_balance(R):
    b = R['bld']; dt = R['dt']
    Ta, Tm = R['Ta'], R['Tm']; To = R['Tdb']; qs = R['q_sens']; Qa, Qm = R['Qa'], R['Qm']
    def flux(Ta_, Tm_, k):
        return (b.Gao * (To[k] - Ta_) + b.Gmo * (To[k] - Tm_) + Qa[k] + Qm[k] - qs[k])
    net = 0.0; thr = 0.0
    for k in range(len(Ta) - 1):
        f = 0.5 * (flux(Ta[k], Tm[k], k) + flux(Ta[k + 1], Tm[k + 1], k))
        net += f * dt
        thr += dt * (abs(b.Gao * (To[k] - Ta[k])) + abs(b.Gmo * (To[k] - Tm[k])) + Qa[k] + Qm[k] + qs[k])
    stored = b.Ca * (Ta[-1] - Ta[0]) + b.Cm * (Tm[-1] - Tm[0])
    return float(abs(net - stored) / thr * 100.0), float(thr)


def timestep_study(strategy='baseline', dts=(5, 15, 30, 60)):
    out = {}
    for m in dts:
        k = simulate(strategy, dt_h=m / 60.0)['kpi']
        out[m] = dict(kwh=k['kwh'], peak=k['peak_kw'], cost=k['cost'])
    ref = out[5]
    for m in dts:
        out[m]['d_energy_pct'] = 100 * (out[m]['kwh'] / ref['kwh'] - 1)
        out[m]['d_peak_pct'] = 100 * (out[m]['peak'] / ref['peak'] - 1)
    return out


def benchmarks(Rb, plant):
    k = Rb['kpi']; area = Rb['bld'].area
    rows = [
        ('Seasonal plant COP', k['cop'], 4.0, 6.0, ''),
        ('Peak cooling density (W/m2)', k['peak_coil'] * 1000 / area, 50.0, 120.0, ''),
        ('Auxiliary share of plant electricity', k['aux_share'], 0.10, 0.25, ''),
        ('Mean tower approach (K)', k['t_approach_mean'], 3.0, 5.0, ''),
        ('Cooling-plant electricity (kWh/m2/day)', k['kwh'] / Rb['days'] / area, 0.06, 0.15, ''),
    ]
    return [dict(name=n, value=float(v), lo=lo, hi=hi, ok=bool(lo <= v <= hi)) for n, v, lo, hi, _ in rows]


PARAMS = ['Ca', 'Cm', 'Gao', 'Gam', 'Gmo', 'int', 'sol', 'latent']


def _bp(name, f, base):
    if name in ('Ca', 'Cm'): return {name: getattr(base, name) * f}
    if name in ('Gao', 'Gam', 'Gmo', 'latent'): return {name: getattr(base, name) * f}
    if name == 'int': return dict(q_int_occ=base.q_int_occ * f, q_int_base=base.q_int_base * f)
    if name == 'sol': return dict(q_sol_air=base.q_sol_air * f, q_sol_mass=base.q_sol_mass * f)


def sensitivity(levels=(0.75, 1.25)):
    base = Building()
    Rb0 = simulate('baseline')['kpi']; Rp0 = simulate('precool')['kpi']
    gap0 = 100 * (Rp0['kwh'] / Rb0['kwh'] - 1); gapc0 = 100 * (Rp0['cost'] / Rb0['cost'] - 1)
    rows = []
    for p in PARAMS:
        for f in levels:
            bp = _bp(p, f, base)
            kb = simulate('baseline', bp=bp)['kpi']; kp = simulate('precool', bp=bp)['kpi']
            rows.append(dict(param=p, f=f, kwh=kb['kwh'], d_abs=100 * (kb['kwh'] / Rb0['kwh'] - 1),
                             gap=100 * (kp['kwh'] / kb['kwh'] - 1), gap_cost=100 * (kp['cost'] / kb['cost'] - 1)))
    return dict(rows=rows, gap0=gap0, gapc0=gapc0, kwh0=Rb0['kwh'])


def ashrae_g14(measured, simulated, n_params=1):
    """NMBE and CV(RMSE) in percent (hourly criteria: |NMBE| <= 10, CV(RMSE) <= 30)."""
    m = np.asarray(measured, float); s = np.asarray(simulated, float); n = len(m)
    nmbe = 100 * np.sum(m - s) / ((n - n_params) * np.mean(m))
    cv = 100 * np.sqrt(np.sum((m - s) ** 2) / (n - n_params)) / np.mean(m)
    return float(nmbe), float(cv)
