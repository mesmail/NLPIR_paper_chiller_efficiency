"""Quasi-static analysis of how model and forecast errors propagate into the choice of running chillers."""
import numpy as np
from chiller import Plant
from staging import all_n

NOMINAL_ABC = (0.30, 0.23, 0.47)


def abc_from_a(a):
    f = (1.0 - a) / 0.70
    return (a, 0.23 * f, 0.47 * f)


def configure(plant, k=1.0, abc=NOMINAL_ABC):
    plant.cop_ref = plant.cop_ref0 / k
    plant.eirfplr = tuple(abc)


def on_steps(R):
    return np.where((R['n'] > 0) & (R['q_coil'] > 0.5))[0]


def headroom(R):
    """Energy of the rule-based staging actually used vs the best fixed stage count at each step (quasi-static)."""
    plant = Plant(); rule = opt = 0.0; diff = 0; idx = on_steps(R)
    q_l, nr_l, no_l, pr_l, po_l, nm_l = [], [], [], [], [], []
    for i in idx:
        v = all_n(plant, R['q_coil'][i], R['Twb'][i])
        n_rule = int(R['n'][i]); n_opt = int(np.argmin(v)) + 1
        rule += v[n_rule - 1]; opt += v[n_opt - 1]; diff += (n_rule != n_opt)
        q_l.append(R['q_coil'][i]); nr_l.append(n_rule); no_l.append(n_opt); pr_l.append(v[n_rule - 1]); po_l.append(v[n_opt - 1])
        nm_l.append(int(np.argmax(np.isfinite(v))) + 1)            # minimum feasible stage count
    nr, no, pr, po, nm = (np.array(x) for x in (nr_l, no_l, pr_l, po_l, nm_l))
    loss = pr - po; mis = nr != no
    k10 = int(round(0.10 * len(loss)))
    switches = lambda s: int(np.sum(s[1:] != s[:-1]))
    return dict(saving_pct=100 * (1 - opt / rule), steps_differ_pct=100 * diff / len(idx), n_steps=len(idx),
                rule_kwh=rule * R['dt'], opt_kwh=opt * R['dt'],
                # --- diagnostics: direction, concentration and switching cost of the headroom ---
                rule_too_many_pct=100 * float(np.mean(nr > no)),   # rule runs MORE chillers than optimal
                rule_too_few_pct=100 * float(np.mean(nr < no)),    # ... or FEWER
                opt_is_min_feasible_pct=100 * float(np.mean(no == nm)),
                loss_share_worst_decile_pct=100 * float(np.sort(loss)[::-1][:k10].sum() / loss.sum()),
                median_mismatch_waste_pct=100 * float(np.median(loss[mis] / pr[mis])),
                switches_rule=switches(nr), switches_opt=switches(no),
                stage_share_pct={n: 100 * float(np.mean(nr == n)) for n in (1, 2, 3)},
                median_plr=float(np.median(R['plr'][idx])),
                _q=q_l and np.array(q_l), _n_rule=nr, _n_opt=no, _p_rule=pr, _p_opt=po)


def decision_study(R, models):
    """models: name -> function(i) -> (k, abc). Returns, per model, the share of steps where the chosen stage count
    differs from the optimum under the true plant, and the extra energy that results."""
    plant_t, plant_m = Plant(), Plant()
    idx = on_steps(R)
    res = {m: dict(mismatch=0, e=0.0, min_feas=0) for m in models}; res['rule'] = dict(mismatch=0, e=0.0, min_feas=0)
    e_star = 0.0
    for i in idx:
        q, tw = R['q_coil'][i], R['Twb'][i]
        configure(plant_t, R['k_true'][i], abc_from_a(R['a_true'][i]))
        vt = all_n(plant_t, q, tw); ns = int(np.argmin(vt)); e_star += vt[ns]
        n_min = int(np.argmax(np.isfinite(vt)))
        for name, fn in models.items():
            k, abc = fn(i); configure(plant_m, k, abc)
            vm = all_n(plant_m, q, tw); nm = int(np.argmin(vm))
            if not np.isfinite(vt[nm]): nm = ns
            res[name]['mismatch'] += (nm != ns); res[name]['e'] += vt[nm]
            res[name]['min_feas'] += (nm == n_min)      # did the model just pick the fewest feasible units?
        nr = int(R['n'][i]) - 1
        res['rule']['mismatch'] += (nr != ns); res['rule']['e'] += vt[nr]
        res['rule']['min_feas'] += (nr == n_min)
    for name in res:
        res[name] = dict(mismatch_pct=100 * res[name]['mismatch'] / len(idx), excess_pct=100 * (res[name]['e'] / e_star - 1),
                         picks_min_feasible_pct=100 * res[name]['min_feas'] / len(idx))
    res['n_steps'] = len(idx)
    return res


def model_error_sweep(R, ks=(0.8, 0.9, 1.1, 1.2, 1.3), a_values=(0.15, 0.22, 0.38, 0.45, 0.55)):
    """Stage-choice loss when the controller's model is wrong in scale (k) or in part-load shape (a), true plant nominal."""
    out = {'k': {}, 'a': {}}
    for k in ks:
        r = decision_study(R, {'m': lambda i, k=k: (k, NOMINAL_ABC)})
        out['k'][k] = r['m']
    for a in a_values:
        r = decision_study(R, {'m': lambda i, a=a: (1.0, abc_from_a(a))})
        out['a'][a] = r['m']
    return out


def forecast_decision(R, target, q_hat):
    """Stage-choice loss when the staging decision is taken on a forecast load q_hat but the true load q arrives."""
    plant = Plant(); loss = star = 0.0; mism = 0
    for t, qh in zip(target, q_hat):
        q, tw = R['q_coil'][t], R['Twb'][t]
        vt = all_n(plant, q, tw); ns = int(np.argmin(vt)); star += vt[ns]
        vh = all_n(plant, max(qh, 1.0), tw); nh = int(np.argmin(vh))
        nmin = int(np.argmax(np.isfinite(vt)))
        nh = max(nh, nmin)
        loss += vt[nh]; mism += (nh != ns)
    return dict(mismatch_pct=100 * mism / len(target), excess_pct=100 * (loss / star - 1))


def forecast_error_sweep(R, sigmas=(0.0, 0.05, 0.10, 0.20, 0.30, 0.40), seeds=(0, 1, 2), subsample=2500):
    idx = on_steps(R); rng0 = np.random.default_rng(5)
    sel = rng0.choice(idx, size=min(subsample, len(idx)), replace=False)
    out = {}
    for s in sigmas:
        rs = []
        for sd in seeds:
            rng = np.random.default_rng(100 + sd)
            qh = R['q_coil'][sel] * (1 + s * rng.standard_normal(len(sel)))
            rs.append(forecast_decision(R, sel, qh))
        out[s] = dict(mismatch_pct=float(np.mean([r['mismatch_pct'] for r in rs])),
                      excess_pct=float(np.mean([r['excess_pct'] for r in rs])))
    return out
