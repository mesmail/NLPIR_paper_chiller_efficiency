"""Short-horizon cooling-load forecasting with walk-forward evaluation.

Target: coil load (kW) H steps ahead (15-minute steps), evaluated only when the plant is scheduled on.
Features: time of day, occupancy schedule, current and lagged load, zone temperature, current weather,
the same time yesterday, and a (noisy) weather forecast for the target time."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STEPS_PER_DAY = 96
# weather-forecast error (standard deviation) by horizon in steps: dry-bulb K, solar relative, RH %
FCST_ERR = {1: (0.3, 0.05, 1.5), 4: (0.6, 0.10, 3.0), 16: (1.2, 0.25, 6.0)}
TRAIN_DAYS0, FOLD_DAYS = 28, 7


def build(R, H, forecast='noisy', seed=3, use_weather_fcst=True):
    rng = np.random.default_rng(seed + H)
    N = len(R['t']); q, Ta, Tdb, RH, G, h, occ = (R[k] for k in ('q_coil', 'Ta', 'Tdb', 'RH', 'G', 'h', 'occ'))
    sT, sG, sR = FCST_ERR[H] if forecast == 'noisy' else (0.0, 0.0, 0.0)
    eT = rng.normal(0, sT, N); eG = rng.normal(0, sG, N); eR = rng.normal(0, sR, N)
    j = np.arange(4 + 0, N - H)                       # origin index i, target index i + H
    j = j[(j + H >= STEPS_PER_DAY) & (h[j + H] >= 6.0) & (h[j + H] < 18.5)]
    t = j + H
    cols = [np.sin(2 * np.pi * h[t] / 24), np.cos(2 * np.pi * h[t] / 24), occ[t],
            q[j], q[j - 1], q[j - 2], q[j - 4], Ta[j], Tdb[j], RH[j], G[j], q[t - STEPS_PER_DAY]]
    names = ['sin_h', 'cos_h', 'occ_t', 'q_0', 'q_1', 'q_2', 'q_4', 'Ta_0', 'Tdb_0', 'RH_0', 'G_0', 'q_yday']
    if use_weather_fcst:
        cols += [Tdb[t] + eT[t], np.clip(G[t] * (1 + eG[t]), 0, None), np.clip(RH[t] + eR[t], 0, 100)]
        names += ['Tdb_f', 'G_f', 'RH_f']
    X = np.column_stack(cols)
    return dict(X=X, y=q[t], origin=j, target=t, day=t // STEPS_PER_DAY, names=names,
                persist=q[j], seasonal=q[t - STEPS_PER_DAY])


def _models():
    return {
        'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'GradientBoosting': HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=15,
                                                          min_samples_leaf=20, l2_regularization=1.0, random_state=0),
    }


def walk_forward(D, last_day):
    day = D['day']; preds = {k: np.full(len(day), np.nan) for k in ('GradientBoosting', 'Ridge')}
    folds = []
    d0 = TRAIN_DAYS0
    while d0 <= last_day:
        te = (day >= d0) & (day < d0 + FOLD_DAYS); tr = day < d0
        if te.any() and tr.sum() > 200:
            for name, m in _models().items():
                m.fit(D['X'][tr], D['y'][tr]); preds[name][te] = m.predict(D['X'][te])
            folds.append((d0, te))
        d0 += FOLD_DAYS
    return preds, folds


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    ok = ~np.isnan(p)
    e = p[ok] - y[ok]
    return dict(mae=float(np.mean(abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                cv_rmse=float(100 * np.sqrt(np.mean(e ** 2)) / np.mean(y[ok])), n=int(ok.sum()))


def run(R, horizons=(1, 4, 16)):
    last_day = int(R['t'][-1] // 24)
    out = {}; store = {}
    for H in horizons:
        out[H] = {}
        for label, fc, uw in (('noisy', 'noisy', True), ('perfect', 'perfect', True), ('no_wx_fcst', 'noisy', False)):
            D = build(R, H, forecast=fc, use_weather_fcst=uw)
            preds, folds = walk_forward(D, last_day)
            ev = ~np.isnan(preds['GradientBoosting'])
            res = {m: metrics(D['y'], np.where(ev, preds[m], np.nan)) for m in ('GradientBoosting', 'Ridge')}
            res['Persistence'] = metrics(D['y'], np.where(ev, D['persist'], np.nan))
            res['SeasonalNaive'] = metrics(D['y'], np.where(ev, D['seasonal'], np.nan))
            res['skill_vs_persistence'] = 1 - res['GradientBoosting']['rmse'] / res['Persistence']['rmse']
            res['fold_cv_rmse'] = [metrics(D['y'], np.where(te, preds['GradientBoosting'], np.nan))['cv_rmse'] for _, te in folds]
            out[H][label] = res
            store[(H, label)] = dict(D=D, preds=preds, ev=ev)
    return out, store
