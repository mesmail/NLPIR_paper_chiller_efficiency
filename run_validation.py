"""Reproduces every verification and validation result and Figures 3 and 4."""
import json, numpy as np
from style import *
from simulate import simulate
from chiller import Plant
import validation as V


def main():
    Rb = simulate('baseline'); Rp = simulate('precool'); plant = Plant()
    res = {}
    res['integrator_err_K'] = V.integrator_vs_analytical(Rb)
    res['balance_pct'], res['balance_throughput_kwh'] = V.energy_balance(Rb)
    res['timestep'] = V.timestep_study('baseline')
    res['timestep_precool'] = V.timestep_study('precool')
    res['ahri'] = plant.ahri()
    res['benchmarks'] = V.benchmarks(Rb, plant)
    res['sens'] = V.sensitivity()
    res['taus'] = [float(t) for t in Rb['bld'].time_constants()]
    res['peak_hour_base'] = Rb['kpi']['peak_hour']
    # morning pull-down peak vs afternoon peak (baseline)
    # ---- COP surface (Figure 4) ----
    twbs = np.linspace(20, 28, 33); cap_tot = 3 * plant.cap
    loads = np.linspace(8, 3 * plant.cap_unit(30) * 0.999, 400)
    Z = np.array([plant.plant_cop_curve(loads, tw) for tw in twbs])
    res['cop_curve_tw25'] = dict(load=loads.tolist(), cop=plant.plant_cop_curve(loads, 25.0).tolist())
    c25 = np.array(res['cop_curve_tw25']['cop'])
    # local maxima just before each staging threshold
    peaks = []
    for n in (1, 2):
        qthr = plant.up * n * plant.cap_unit(25 + 4.5)
        i = int(np.argmin(abs(loads - qthr))); lo = max(0, i - 3)
        peaks.append(dict(load_kw=float(loads[i]), frac=float(loads[i] / cap_tot), cop_before=float(np.nanmax(c25[lo:i + 1])),
                          cop_after=float(c25[i + 6])))
    res['stage_peaks'] = peaks
    res['cop_range'] = [float(np.nanmin(Z)), float(np.nanmax(Z))]
    res['plant_cap_kw'] = cap_tot
    json.dump(res, open('results_validation.json', 'w'), indent=1, default=float)
    # ---------------- Figure 4 ----------------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a = ax[0]; im = a.pcolormesh(loads / cap_tot * 100, twbs, Z, shading='auto', cmap='viridis')
    a.set_xlabel('plant load (% of 450 kW nominal capacity)'); a.set_ylabel('ambient wet-bulb (C)'); a.set_title('(a) Plant COP surface'); a.grid(False)
    cb = fig.colorbar(im, ax=a, pad=0.02); cb.set_label('plant COP (incl. pumps, fans)')
    a = ax[1]
    for tw, c in ((22, '#2a9d8f'), (25, C_BASE), (27, C_PRE)):
        a.plot(loads / cap_tot * 100, plant.plant_cop_curve(loads, tw), color=c, lw=1.4, label=f'wet-bulb {tw} C')
    for p in peaks:
        a.axvline(p['frac'] * 100, color='#999', lw=0.7, ls=':')
    a.set_xlabel('plant load (% of nominal capacity)'); a.set_ylabel('plant COP'); a.set_title('(b) Cross-section: staging discontinuities'); a.legend(loc='lower right')
    fig.tight_layout(); fig.savefig('figs/fig4_cop_surface.png'); plt.close(fig)
    # ---------------- Figure 3 ----------------
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    # (a) integrator error
    b = Rb['bld']; dt = Rb['dt']; n = int(24 / dt); s = 3 * n
    u = np.column_stack([Rb['Tdb'][s:s + n], Rb['Qa'][s:s + n], Rb['Qm'][s:s + n], Rb['q_sens'][s:s + n]])
    xr = np.array([Rb['Ta'][s], Rb['Tm'][s]]); xe = xr.copy(); er = []; tt = []
    for k in range(n):
        xr = b.step_rk4(xr, u[k], dt); xe = b.step_exact(xe, u[k], dt); er.append(abs(xr[0] - xe[0])); tt.append((k + 1) * dt)
    a = ax[0, 0]; a.plot(tt, np.array(er) * 1e5, color=C_BASE); a.set_xlabel('hours'); a.set_ylabel('|RK4 - exact| (1e-5 K)')
    a.set_title('(a) Integrator vs matrix-exponential solution'); a.set_xlim(0, 24)
    a.text(0.97, 0.9, f"max error {res['integrator_err_K']:.1e} K", transform=a.transAxes, ha='right')
    # (b) time step
    a = ax[0, 1]; ms = [5, 15, 30, 60]; x = np.arange(4)
    a.bar(x - 0.18, [res['timestep'][m]['d_energy_pct'] for m in ms], 0.36, color=C_BASE, label='energy')
    a.bar(x + 0.18, [res['timestep'][m]['d_peak_pct'] for m in ms], 0.36, color=C_PRE, label='peak demand')
    a.set_xticks(x); a.set_xticklabels([f'{m} min' for m in ms]); a.axhline(0, color='k', lw=0.6)
    a.set_ylabel('difference vs 5-min solution (%)'); a.set_title('(b) Time-step independence'); a.legend(loc='lower left')
    # (c) part load
    a = ax[1, 0]; plr = np.linspace(0.12, 1, 60)
    for tcw, c, lab in ((29.44, C_BASE, 'ECW 29.4 C (85 F)'), (23.89, C_ACC, 'ECW 23.9 C (75 F)'), (18.33, C_PRE, 'ECW 18.3 C (65 F)')):
        a.plot(plr * 100, [3.517 / plant.chiller_cop_at(p, tcw) for p in plr], color=c, label=lab)
    for (p, tcw, w), cp in zip(res['ahri']['pts'], res['ahri']['cops']):
        a.plot(p * 100, 3.517 / cp, 'ko', ms=4)
    a.set_xlabel('part-load ratio (%)'); a.set_ylabel('chiller kW/ton'); a.set_ylim(0.4, 1.0)
    a.set_title(f"(c) AHRI 550/590 points: {res['ahri']['full_kw_ton']:.3f} full-load, IPLV {res['ahri']['iplv_kw_ton']:.3f}"); a.legend(loc='upper right')
    # (d) sensitivity
    a = ax[1, 1]; rows = res['sens']['rows']; params = V.PARAMS; lab = {'Ca': 'C air', 'Cm': 'C mass', 'Gao': 'G air-out', 'Gam': 'G air-mass', 'Gmo': 'G mass-out', 'int': 'internal', 'sol': 'solar', 'latent': 'latent'}
    y = np.arange(len(params))
    lo = [min(r['d_abs'] for r in rows if r['param'] == p) for p in params]; hi = [max(r['d_abs'] for r in rows if r['param'] == p) for p in params]
    a.barh(y, hi, color=C_BASE, alpha=0.8, label='absolute energy (+/-25 % parameter)'); a.barh(y, lo, color=C_BASE, alpha=0.8)
    gsh = [max(abs(r['gap'] - res['sens']['gap0']) for r in rows if r['param'] == p) for p in params]
    a.scatter([0] * len(params), y, s=0)
    a.plot([g for g in gsh], y, 'D', color=C_PRE, ms=4, label='shift in strategy gap (pp)')
    a.set_yticks(y); a.set_yticklabels([lab[p] for p in params]); a.invert_yaxis(); a.axvline(0, color='k', lw=0.6)
    a.set_xlabel('bars: % energy change   diamonds: gap shift (pp)', fontsize=8); a.set_title('(d) Sensitivity: absolute vs relative'); a.legend(loc='upper center', bbox_to_anchor=(0.42, -0.22), fontsize=7, ncol=1)
    fig.tight_layout(); fig.savefig('figs/fig3_verification.png'); plt.close(fig)
    print(json.dumps({k: v for k, v in res.items() if k not in ('cop_curve_tw25', 'sens', 'timestep_precool')}, indent=1, default=float))
    sr = res['sens']['rows']
    print('abs range', min(r['d_abs'] for r in sr), max(r['d_abs'] for r in sr))
    print('gap range', min(r['gap'] for r in sr), max(r['gap'] for r in sr), 'nominal', res['sens']['gap0'], 'max shift', max(abs(r['gap'] - res['sens']['gap0']) for r in sr))
    print('costgap range', min(r['gap_cost'] for r in sr), max(r['gap_cost'] for r in sr), 'nominal', res['sens']['gapc0'])
    print('tsp', {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in res['timestep_precool'].items()})


if __name__ == '__main__':
    main()
