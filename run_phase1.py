"""Reproduces the testbed baseline results and Figures 2 and 5."""
import json, numpy as np
from style import *
from simulate import simulate


def hourly_profile(R, key):
    h = R['h']; v = R[key]; out = np.zeros(int(24 / R['dt']))
    idx = np.round(h / R['dt']).astype(int) % len(out)
    for i in range(len(out)): out[i] = v[idx == i].mean()
    return np.arange(len(out)) * R['dt'], out


def main():
    Rb = simulate('baseline'); Rp = simulate('precool')
    kb, kp = Rb['kpi'], Rp['kpi']
    json.dump(dict(baseline=kb, precool=kp), open('results_phase1.json', 'w'), indent=1)
    # ---------------- Figure 2 ----------------
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
    names = ['Baseline\nschedule + PI', 'Rule-based\npre-cooling']; cols = [C_BASE, C_PRE]
    a = ax[0, 0]; a.bar(names, [kb['kwh'], kp['kwh']], color=cols, width=0.55)
    a.set_title('(a) Electricity, 14 days (kWh)'); a.set_ylim(0, max(kb['kwh'], kp['kwh']) * 1.15)
    for i, v in enumerate([kb['kwh'], kp['kwh']]): a.text(i, v * 1.02, f'{v:,.0f}', ha='center')
    a = ax[0, 1]
    a.bar(names, [kb['cost_energy'], kp['cost_energy']], color=cols, width=0.55, label='energy charge')
    a.bar(names, [kb['cost_md'], kp['cost_md']], bottom=[kb['cost_energy'], kp['cost_energy']], color=cols, alpha=0.45, width=0.55, hatch='//', label='demand charge')
    a.set_title('(b) Cost, 14 days (RM, placeholder tariff)'); a.legend(loc='upper center', ncol=2)
    for i, v in enumerate([kb['cost'], kp['cost']]): a.text(i, v * 1.01, f'{v:,.0f}', ha='center')
    a.set_ylim(0, max(kb['cost'], kp['cost']) * 1.3)
    a = ax[1, 0]
    for R, c, n in ((Rb, C_BASE, 'Baseline'), (Rp, C_PRE, 'Pre-cooling')):
        x, y = hourly_profile(R, 'P'); a.plot(x, y, color=c, lw=1.6, label=n)
    a.axvspan(*PEAK, color='#cccccc', alpha=0.4, lw=0); a.text(14.3, a.get_ylim()[1] * 0.92, 'peak tariff', fontsize=7)
    a.set_xlim(0, 24); a.set_xticks(range(0, 25, 4)); a.set_xlabel('hour of day'); a.set_ylabel('mean electrical demand (kW)')
    a.set_title('(c) Average daily demand profile'); a.legend(loc='upper left')
    a = ax[1, 1]
    for R, c, n in ((Rb, C_BASE, 'Baseline'), (Rp, C_PRE, 'Pre-cooling')):
        x, y = hourly_profile(R, 'Ta'); a.plot(x, y, color=c, lw=1.6, label=n)
    a.axhspan(22, 24.5, color=C_ACC, alpha=0.12, lw=0); a.text(0.3, 24.55, 'comfort band 22-24.5 C', fontsize=7, color=C_ACC)
    a.set_xlim(0, 24); a.set_xticks(range(0, 25, 4)); a.set_xlabel('hour of day'); a.set_ylabel('zone air temperature (C)')
    a.set_title('(d) Average daily zone temperature'); a.legend(loc='lower right')
    fig.tight_layout(); fig.savefig('figs/fig2_strategies.png'); plt.close(fig)
    # ---------------- Figure 5 ----------------
    d0, d1 = 3, 6
    m = (Rb['t'] >= d0 * 24) & (Rb['t'] < d1 * 24); t = Rb['t'][m] - d0 * 24
    fig, ax = plt.subplots(4, 1, figsize=(7.2, 8.0), sharex=True)
    a = ax[0]; a.plot(t, Rb['Tdb'][m], color='#c0392b', label='dry-bulb'); a.plot(t, Rb['Twb'][m], color='#2980b9', label='wet-bulb')
    a.set_ylim(21, 35); a.set_ylabel('temperature (C)'); a2 = a.twinx(); a2.fill_between(t, Rb['G'][m], color='#f1c40f', alpha=0.3, lw=0); a2.set_ylabel('solar (W/m2)'); a2.grid(False)
    a2.spines['right'].set_visible(True); a.legend(loc='upper left', ncol=2); a.set_title('(a) Boundary conditions')
    a = ax[1]; a.plot(t, Rb['Ta'][m], color=C_BASE, label='zone air'); a.plot(t, Rb['Tm'][m], color=C_GREY, lw=1, label='structural mass')
    a.axhspan(22, 24.5, color=C_ACC, alpha=0.12, lw=0); a.set_ylim(21.5, 28.5); a.set_ylabel('temperature (C)'); a.legend(loc='upper left', ncol=2); a.set_title('(b) Zone response (band 22-24.5 C shaded)')
    a = ax[2]; a.plot(t, Rb['q_coil'][m], color=C_BASE, label='coil load (incl. latent)'); a.plot(t, Rb['Qa'][m] + Rb['Qm'][m], color=C_GREY, lw=1, label='internal + solar gains')
    a.set_ylim(-10, 470); a.set_ylabel('thermal (kW)'); a.legend(loc='upper left', ncol=2); a.set_title('(c) Thermal loads')
    a = ax[3]; a.plot(t, Rb['P'][m], color=C_PRE, label='plant electrical demand'); a.set_ylim(0, 105); a.set_ylabel('electrical (kW)')
    a2 = a.twinx(); a2.step(t, Rb['n'][m], where='post', color='#333', lw=1, label='chillers running'); a2.set_ylim(0, 4); a2.set_yticks([0, 1, 2, 3]); a2.set_ylabel('chillers on'); a2.grid(False)
    a2.spines['right'].set_visible(True); a.set_xlabel('hours from start of day 1'); a.set_title('(d) Electrical demand with chiller staging')
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, loc='upper left', ncol=2)
    a.set_xticks(range(0, 73, 6)); a.set_xlim(0, 72)
    for aa in ax: aa.axvline(24, color='#999', lw=0.6, ls=':'); aa.axvline(48, color='#999', lw=0.6, ls=':')
    fig.tight_layout(); fig.savefig('figs/fig5_testbed.png'); plt.close(fig)
    print(json.dumps(dict(baseline={k: round(v, 3) for k, v in kb.items()}, precool={k: round(v, 3) for k, v in kp.items()}), indent=1))


if __name__ == '__main__':
    main()
