"""Figure 1 - co-simulation architecture."""
from style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, w, h, text, dashed=False, fc='#e8f0f7', ec=C_BASE):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc='white' if dashed else fc,
                       ec=ec, lw=1.4, ls='--' if dashed else '-')
    ax.add_patch(p); ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=8, color='black' if not dashed else '#555')


def arrow(ax, a, b, dashed=False, label=None, rad=0.0, lo=(0, 0.12)):
    ar = FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=11, lw=1.2, color=C_GREY if dashed else '#333',
                         ls='--' if dashed else '-', connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(ar)
    if label:
        ax.text((a[0] + b[0]) / 2 + lo[0], (a[1] + b[1]) / 2 + lo[1], label, fontsize=7, ha='center', color='#333')


def main(path='figs/fig1_architecture.png'):
    fig, ax = plt.subplots(figsize=(7.2, 4.0)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off'); ax.grid(False)
    box(ax, 0.2, 4.1, 2.2, 1.2, 'Weather &\noccupancy\n(tropical diurnal, storms)')
    box(ax, 3.6, 4.1, 2.4, 1.2, 'Building\n3R2C thermal model\n(state-space export)')
    box(ax, 7.2, 4.1, 2.4, 1.2, 'Chiller plant\n3 x 150 kW DOE-2 EIR,\ntower, pumps, staging')
    box(ax, 3.6, 2.0, 2.4, 1.1, 'Controller\nscheduled PI zone loop\n(replaced in Phase 3)')
    box(ax, 7.2, 2.0, 2.4, 1.1, 'Tariff\ntime-of-use + max demand')
    box(ax, 7.2, 0.2, 2.4, 1.0, 'KPIs\nkWh, cost, peak, COP, comfort')
    box(ax, 0.2, 2.0, 2.4, 1.1, 'Load forecaster\n(Phase 2)', dashed=True, ec=C_GREY)
    box(ax, 0.2, 0.2, 2.4, 1.1, 'Online COP estimator\n(Phase 2)', dashed=True, ec=C_GREY)
    box(ax, 3.6, 0.2, 2.4, 1.1, 'Predictive controller\nMPC + staging (Phase 3)', dashed=True, ec=C_GREY)
    arrow(ax, (2.4, 4.7), (3.6, 4.7), label='T_out, solar, occ.')
    arrow(ax, (6.0, 4.9), (7.2, 4.9), label='coil load', lo=(0, 0.12))
    arrow(ax, (7.2, 4.4), (6.0, 4.4), label='cooling delivered', lo=(0.05, -0.34))
    arrow(ax, (4.8, 3.1), (4.8, 4.1), label='cooling\ncommand', lo=(0.65, -0.1))
    arrow(ax, (3.9, 4.1), (3.9, 3.1), label='T_zone', lo=(-0.45, 0.0))
    arrow(ax, (8.4, 4.1), (8.4, 3.1), label='power', lo=(0.45, 0.0))
    arrow(ax, (8.4, 2.0), (8.4, 1.2))
    arrow(ax, (2.6, 2.3), (3.7, 1.2), dashed=True)
    arrow(ax, (2.6, 0.75), (3.6, 0.75), dashed=True)
    arrow(ax, (4.8, 1.3), (4.8, 2.0), dashed=True, label='swap', lo=(0.35, 0))
    ax.set_title('Co-simulation architecture (15-minute step)', loc='left')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == '__main__':
    main()
