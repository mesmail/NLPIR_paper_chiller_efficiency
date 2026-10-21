"""Co-simulation loop (weather -> building -> plant -> tariff) and KPI calculation."""
import numpy as np
from weather import make_weather, resample
from building import Building
from chiller import Plant
import tariff
import controllers

SPIN_DAYS = 2


def simulate(strategy='baseline', bp=None, pp=None, days=14, dt_h=0.25, seed=7, integrator='rk4', drift=None, plr_drift=None):
    b = Building(**(bp or {})); plant = Plant(**(pp or {}))
    w = resample(make_weather(days + SPIN_DAYS, seed), dt_h)
    ctrl = controllers.make(strategy)
    N = len(w['t']); x = np.array([26.0, 26.5])
    keys = ['k_true', 'a_true', 'Ta', 'Tm', 'q_sens', 'q_coil', 'n', 'P_ch', 'P_cwp', 'P_fan', 'P_chwp', 'P', 't_cw', 'plr', 'Qa', 'Qm', 'sp']
    R = {k: np.zeros(N) for k in keys}
    for k in range(N):
        h = w['h'][k]
        if drift is not None:
            plant.cop_ref = plant.cop_ref0 * drift(w['t'][k] - SPIN_DAYS * 24.0)
        if plr_drift is not None:
            plant.eirfplr = plr_drift(w['t'][k] - SPIN_DAYS * 24.0)
        R['k_true'][k] = plant.cop_ref0 / plant.cop_ref
        R['a_true'][k] = plant.eirfplr[0]
        qs_cmd = ctrl.command(x[0], h, dt_h)
        on, sp = ctrl.schedule(h)
        r = plant.operate(b.latent * qs_cmd, w['Twb'][k])
        qs = r['q'] / b.latent
        Qa, Qm = b.gains(w['G'][k], w['occ'][k])
        u = np.array([w['Tdb'][k], Qa, Qm, qs])
        R['Ta'][k], R['Tm'][k] = x
        R['q_sens'][k], R['q_coil'][k], R['n'][k] = qs, r['q'], r['n']
        for kk in ('P_ch', 'P_cwp', 'P_fan', 'P_chwp', 't_cw', 'plr'): R[kk][k] = r[kk]
        R['P'][k] = plant.p_total(r); R['Qa'][k], R['Qm'][k] = Qa, Qm; R['sp'][k] = sp
        x = b.step_rk4(x, u, dt_h) if integrator == 'rk4' else b.step_exact(x, u, dt_h)
    s0 = int(round(SPIN_DAYS * 24 / dt_h))
    for kk in R: R[kk] = R[kk][s0:]
    for kk in ('t', 'h', 'Tdb', 'Twb', 'G', 'occ', 'RH'):
        R[kk] = w[kk][s0:]
    R['t'] = R['t'] - R['t'][0]
    R['dt'] = dt_h; R['days'] = days; R['bld'] = b
    R['kpi'] = kpis(R, days, dt_h)
    return R


def kpis(R, days, dt_h):
    e = R['P'] * dt_h
    peak = float(R['P'].max())
    total, e_cost, md_cost = tariff.cost(e, R['h'], peak, days)
    Ta, occ = R['Ta'], R['occ']
    dev = np.where(occ > 0.5, np.maximum(0, np.maximum(Ta - 24.5, 22.0 - Ta)), 0.0)
    q_th = float(np.sum(R['q_coil']) * dt_h)
    return dict(kwh=float(e.sum()), cost=total, cost_energy=e_cost, cost_md=md_cost, peak_kw=peak,
                cop=q_th / float(e.sum()), dev_kh=float(dev.sum() * dt_h), q_th_kwh=q_th,
                peak_coil=float(R['q_coil'].max()),
                aux_share=float((R['P_cwp'] + R['P_fan'] + R['P_chwp']).sum() / R['P'].sum()),
                t_approach_mean=float(np.mean((R['t_cw'] - R['Twb'])[R['n'] > 0])),
                peak_hour=float(R['h'][int(np.argmax(R['P']))]))
