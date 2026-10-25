"""Phase 2: load forecasting, online efficiency identification, and the effect of model/forecast error on staging.
Writes results_phase2.json and figures fig6-fig8."""
import json, time
import numpy as np
from style import *
from simulate import simulate
import forecast as F
import analysis as A
from online_cop import identify, identify_curve, fouling_cleaning, part_load_drift

DAYS = 120


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)): return float(o)
    return o


def lag_days(err_daily, start, tol=0.02):
    for d in range(start, len(err_daily)):
        if err_daily[d] < tol: return d - start
    return None


def main():
    t0 = time.time(); res = {}
    R0 = simulate('baseline', days=DAYS)
    RA = simulate('baseline', days=DAYS, drift=fouling_cleaning)
    RB = simulate('baseline', days=DAYS, plr_drift=part_load_drift)
    # ------------------------------------------------ 1. forecasting
    fc, store = F.run(R0)
    res['forecast'] = {str(H * 15): v for H, v in fc.items()}
    res['forecast_setup'] = dict(days=DAYS, train_days0=F.TRAIN_DAYS0, fold_days=F.FOLD_DAYS,
                                 weather_fcst_error={str(H * 15): dict(T_K=v[0], solar_rel=v[1], RH_pct=v[2]) for H, v in F.FCST_ERR.items()})
    # ------------------------------------------------ 2. online identification
    est = {}
    lams = (0.99, 0.998, 1.0)
    for lam in lams:
        est[lam] = identify(RA, lam=lam)
    kt = RA['k_true']; day = (RA['t'] // 24).astype(int)
    def daily(err):
        return np.array([np.nanmean(err[day == d]) for d in range(DAYS)])
    res['online_scalar'] = {}
    for lam in lams:
        err = np.abs(est[lam]['k_hat'] - kt) / kt
        on = ~np.isnan(est[lam]['pred_err'])
        res['online_scalar'][str(lam)] = dict(
            mean_param_err_pct=100 * float(err[on].mean()),
            max_param_err_pct=100 * float(err[on].max()),
            pred_rmse_online_pct=100 * float(np.sqrt(np.nanmean(est[lam]['pred_err'] ** 2))),
            pred_rmse_static_pct=100 * float(np.sqrt(np.nanmean(est[lam]['pred_static_err'] ** 2))))
    ctrl = identify(R0, lam=0.998)
    res['online_scalar']['no_drift_control'] = dict(mean_abs_k_minus_1_pct=100 * float(np.nanmean(np.abs(ctrl['k_hat'][~np.isnan(ctrl['pred_err'])] - 1))))
    noise_levels = (0.005, 0.01, 0.02, 0.05); res['online_scalar']['noise_sensitivity'] = {}
    for lam in (0.99, 0.998):
        res['online_scalar']['noise_sensitivity'][str(lam)] = {}
        for nz in noise_levels:
            e = identify(RA, noise=nz, lam=lam); on = ~np.isnan(e['pred_err'])
            res['online_scalar']['noise_sensitivity'][str(lam)][str(nz)] = 100 * float((np.abs(e['k_hat'] - kt) / kt)[on].mean())
    sb = identify(RB, lam=0.998); cb = identify_curve(RB, lam=0.999)
    onb = ~np.isnan(cb['pred_err'])
    res['online_curve'] = dict(pred_rmse_curve_pct=100 * float(np.sqrt(np.nanmean(cb['pred_err'] ** 2))),
                               pred_rmse_scalar_pct=100 * float(np.sqrt(np.nanmean(sb['pred_err'] ** 2))),
                               pred_rmse_static_pct=100 * float(np.sqrt(np.nanmean(sb['pred_static_err'] ** 2))))
    # ------------------------------------------------ 3. decisions
    hr0 = A.headroom(R0); res['headroom_nominal'] = hr0
    kh = np.roll(est[0.998]['k_hat'], 1); kh[0] = 1.0
    res['decision_fouling'] = A.decision_study(RA, {'static': lambda i: (1.0, A.NOMINAL_ABC), 'online_scalar': lambda i: (kh[i], A.NOMINAL_ABC)})
    khb = np.roll(sb['k_hat'], 1); khb[0] = 1.0
    thb = np.roll(cb['theta'], 1, axis=0); thb[0] = A.NOMINAL_ABC
    res['decision_part_load'] = A.decision_study(RB, {'static': lambda i: (1.0, A.NOMINAL_ABC),
                                                     'online_scalar': lambda i: (khb[i], A.NOMINAL_ABC),
                                                     'online_curve': lambda i: (1.0, tuple(thb[i]))})
    res['model_error_sweep'] = A.model_error_sweep(R0)
    res['forecast_error_sweep'] = A.forecast_error_sweep(R0)
    res['forecast_decision_measured'] = {}
    for H in (1, 4, 16):
        s = store[(H, 'noisy')]; ev = s['ev']
        res['forecast_decision_measured'][str(H * 15)] = A.forecast_decision(R0, s['D']['target'][ev], s['preds']['GradientBoosting'][ev])
    res['runtime_s'] = time.time() - t0
    json.dump(jsonable(res), open('results_phase2.json', 'w'), indent=1)
    # ================================================= figures
    hz = [1, 4, 16]; hl = ['15 min', '1 h', '4 h']
    # ---- Figure 6 forecasting
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    a = ax[0, 0]; models = ['Persistence', 'SeasonalNaive', 'Ridge', 'GradientBoosting']
    cols = [C_GREY, '#b0b0b0', C_ACC, C_BASE]; w = 0.2; x = np.arange(3)
    for k, (m, c) in enumerate(zip(models, cols)):
        a.bar(x + (k - 1.5) * w, [fc[H]['noisy'][m]['cv_rmse'] for H in hz], w, color=c, label={'SeasonalNaive': 'Seasonal naive', 'GradientBoosting': 'Gradient boosting'}.get(m, m))
    a.set_xticks(x); a.set_xticklabels(hl); a.set_yscale('log'); a.set_ylabel('CV(RMSE) (%), log scale'); a.set_xlabel('forecast horizon')
    a.set_ylim(1.5, 400); a.set_title('(a) Walk-forward accuracy by model'); a.legend(fontsize=7, loc='upper left')
    a = ax[0, 1]
    for k, (lab, c, t) in enumerate((('perfect', C_ACC, 'perfect weather forecast'), ('noisy', C_BASE, 'noisy weather forecast'), ('no_wx_fcst', C_PRE, 'no weather forecast'))):
        a.bar(x + (k - 1) * 0.26, [fc[H][lab]['GradientBoosting']['cv_rmse'] for H in hz], 0.26, color=c, label=t)
    a.set_xticks(x); a.set_xticklabels(hl); a.set_ylabel('CV(RMSE) (%)'); a.set_xlabel('forecast horizon'); a.set_title('(b) Value of the weather forecast'); a.legend(fontsize=7, loc='upper left')
    a = ax[1, 0]
    for H, c, lab in ((4, C_BASE, '1 h ahead'), (16, C_PRE, '4 h ahead')):
        s = store[(H, 'noisy')]; D = s['D']; ev = s['ev']; tt = R0['t'][D['target']]
        m = ev & (tt >= 100 * 24) & (tt < 102 * 24)
        if H == 4: a.plot(tt[m] - 100 * 24, D['y'][m], '.', color='k', ms=2.5, label='actual load')
        a.plot(tt[m] - 100 * 24, s['preds']['GradientBoosting'][m], '.', color=c, ms=2.5, label=f'forecast, {lab}')
    a.set_xlabel('hours from start of day 101'); a.set_ylabel('coil load (kW)'); a.set_ylim(100, 540); a.set_title('(c) Two unseen days'); a.legend(fontsize=7, loc='upper right', markerscale=3)
    a = ax[1, 1]
    for H, c, lab in ((1, C_ACC, '15 min'), (4, C_BASE, '1 h'), (16, C_PRE, '4 h')):
        f = fc[H]['noisy']['fold_cv_rmse']; a.plot(F.TRAIN_DAYS0 + F.FOLD_DAYS * np.arange(len(f)), f, 'o-', color=c, ms=3, label=lab)
    a.set_xlabel('first test day of fold'); a.set_ylabel('CV(RMSE) (%)'); a.set_ylim(0, 8.5); a.set_title('(d) Stability across walk-forward folds'); a.legend(fontsize=7, loc='upper center', ncol=3)
    fig.tight_layout(); fig.savefig('figs/fig6_forecasting.png'); plt.close(fig)
    # ---- Figure 7 online identification
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6)); td = RA['t'] / 24
    a = ax[0, 0]; a.plot(td, kt, color='k', lw=1.6, label='true multiplier k')
    for lam, c in zip(lams, (C_PRE, C_BASE, C_ACC)):
        a.plot(td, est[lam]['k_hat'], color=c, lw=1.0, label=f'RLS, forgetting {lam}')
    a.axhline(1.0, color=C_GREY, ls=':', lw=1, label='static model'); a.set_xlabel('day'); a.set_ylabel('chiller power multiplier k'); a.set_title('(a) Tracking fouling and a cleaning event'); a.legend(fontsize=6.5, loc='upper left')
    a = ax[0, 1]; wk = np.arange(0, DAYS, 7)
    def weekly(e): return [100 * np.sqrt(np.nanmean(e[(day >= d) & (day < d + 7)] ** 2)) for d in wk]
    a.plot(wk, weekly(est[0.998]['pred_static_err']), 'o-', color=C_PRE, ms=3, label='static model'); a.plot(wk, weekly(est[0.998]['pred_err']), 'o-', color=C_BASE, ms=3, label='online (RLS)')
    a.axhline(2.0, color=C_GREY, ls=':', lw=1); a.set_ylim(1.4, 9.2); a.text(88, 1.55, 'meter noise 2 %', fontsize=7, color=C_GREY)
    a.set_xlabel('day'); a.set_ylabel('one-step power error, weekly RMSE (%)'); a.set_title('(b) Power prediction error'); a.legend(fontsize=7)
    a = ax[1, 0]; dB = (RB['t'] / 24); i25 = lambda th: th[..., 0] + th[..., 1] * 0.25 + th[..., 2] * 0.0625
    true25 = np.array([i25(np.array(part_load_drift(t * 24))) for t in dB]); nom25 = i25(np.array(A.NOMINAL_ABC))
    a.plot(dB, true25, color='k', lw=1.6, label='true'); a.plot(dB, i25(cb['theta']), color=C_BASE, lw=1.0, label='3-parameter curve RLS')
    a.plot(dB, sb['k_hat'] * nom25, color=C_ACC, lw=1.0, label='scalar RLS'); a.axhline(nom25, color=C_GREY, ls=':', lw=1, label='static model')
    a.set_xlabel('day'); a.set_ylabel('part-load factor at PLR = 0.25'); a.set_title('(c) Tracking a part-load curve change'); a.legend(fontsize=6.5, loc='upper left')
    a = ax[1, 1]; ns = [100 * n for n in noise_levels]
    for lam, c in ((0.99, C_PRE), (0.998, C_BASE)):
        a.plot(ns, [res['online_scalar']['noise_sensitivity'][str(lam)][str(n)] for n in noise_levels], 'o-', color=c, label=f'forgetting {lam}')
    a.set_xlabel('sub-meter noise (%)'); a.set_ylabel('mean |k_hat - k| / k (%)'); a.set_title('(d) Noise vs tracking-lag trade-off'); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig('figs/fig7_online_identification.png'); plt.close(fig)
    # ---- Figure 8 decisions
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    a = ax[0, 0]; fs = res['forecast_error_sweep']; sg = sorted(fs, key=float)
    a.plot([100 * float(s) for s in sg], [fs[s]['excess_pct'] for s in sg], 'o-', color=C_BASE, label='random relative error')
    for H, c, lab in ((1, C_ACC, '15 min'), (4, C_PRE, '1 h'), (16, '#8e44ad', '4 h')):
        cv = fc[H]['noisy']['GradientBoosting']['cv_rmse']; ex = res['forecast_decision_measured'][str(H * 15)]['excess_pct']
        a.plot(cv, ex, 's', color=c, ms=6, label=f'gradient boosting, {lab}')
    a.axhline(hr0['saving_pct'], color='k', ls='--', lw=1); a.text(1, hr0['saving_pct'] + 0.04, 'loss of the rule-based sequencer', fontsize=7)
    a.set_xlabel('load forecast error, relative (%)'); a.set_ylabel('extra plant energy (%)'); a.set_title('(a) Forecast error and staging loss'); a.legend(fontsize=6.5, loc='lower right')
    a = ax[0, 1]; ms_ = res['model_error_sweep']
    labs, vals, cl = [], [], []
    for k in sorted(ms_['k'], key=float): labs.append(f'k x{float(k):.1f}'); vals.append(ms_['k'][k]['excess_pct']); cl.append(C_BASE)
    for aa in sorted(ms_['a'], key=float): labs.append(f'a={float(aa):.2f}'); vals.append(ms_['a'][aa]['excess_pct']); cl.append(C_PRE)
    from matplotlib.patches import Patch
    a.bar(range(len(vals)), vals, color=cl); a.set_xticks(range(len(vals))); a.set_xticklabels(labs, rotation=60, fontsize=7)
    for i_, v_ in enumerate(vals):
        if v_ < 0.02: a.text(i_, 0.04, '<0.02', rotation=90, fontsize=6, ha='center', va='bottom')
    a.set_ylim(0, max(vals) * 1.35)          # headroom so the legend never sits on the tallest bar
    a.legend(handles=[Patch(color=C_BASE, label='wrong scale k'), Patch(color=C_PRE, label='wrong part-load shape a')], fontsize=7, loc='upper left')
    a.set_ylabel('extra plant energy (%)'); a.set_title('(b) Model error in scale vs shape')
    a = ax[1, 0]
    items = [('static', res['decision_part_load']['static']), ('scalar RLS', res['decision_part_load']['online_scalar']),
             ('curve RLS', res['decision_part_load']['online_curve']), ('rule-based\nsequencer', res['decision_part_load']['rule'])]
    a.bar(range(4), [v['excess_pct'] for _, v in items], color=[C_GREY, C_ACC, C_BASE, C_PRE]); a.set_xticks(range(4)); a.set_xticklabels([n for n, _ in items], fontsize=7)
    a.set_ylabel('extra plant energy (%)'); a.set_title('(c) Part-load drift: who chooses the stage count')
    for i, (_, v) in enumerate(items): a.text(i, v['excess_pct'] + 0.05, f"{v['excess_pct']:.2f}", ha='center', fontsize=7)
    a = ax[1, 1]; q, nr, no, pr, po = hr0['_q'], hr0['_n_rule'], hr0['_n_opt'], hr0['_p_rule'], hr0['_p_opt']
    bins = np.arange(0, 100.1, 10); fr = 100 * q / 450.0; idx_ = np.digitize(fr, bins) - 1
    mis = [100 * np.mean(nr[idx_ == b] != no[idx_ == b]) if np.any(idx_ == b) else np.nan for b in range(10)]
    pen = [100 * (pr[idx_ == b].sum() / po[idx_ == b].sum() - 1) if np.any(idx_ == b) else np.nan for b in range(10)]
    a.bar(bins[:-1] + 5, mis, 8, color=C_GREY, label='steps where rule differs from optimum (%)')
    a.plot(bins[:-1] + 5, pen, 'o-', color=C_PRE, label='extra energy (%)')
    a.set_ylim(0, 95); a.set_xlabel('plant load (% of 450 kW)'); a.set_ylabel('%'); a.set_title('(d) Where the rule-based sequencer loses'); a.legend(fontsize=6.5, loc='upper left')
    fig.tight_layout(); fig.savefig('figs/fig8_staging_decisions.png'); plt.close(fig)
    print(json.dumps({k: jsonable(v) for k, v in res.items() if k not in ('forecast', 'forecast_setup')}, indent=1)[:6000])
    print('runtime', round(res['runtime_s'], 1))


if __name__ == '__main__':
    main()
