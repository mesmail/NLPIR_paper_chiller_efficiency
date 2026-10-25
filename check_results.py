"""Checks that a fresh run reproduces the headline values reported in the Phase 1 report."""
import json, sys

P = json.load(open('results_phase1.json'))
V = json.load(open('results_validation.json'))
P2 = json.load(open('results_phase2.json'))
b, p = P['baseline'], P['precool']
s = V['sens']['rows']
g0 = V['sens']['gap0']

checks = [
    # (description, value, expected, tolerance)
    ('Baseline electricity (kWh)',            b['kwh'],                      7665,  1),
    ('Baseline total cost (RM)',              b['cost'],                     3203,  1),
    ('Baseline peak demand (kW)',             b['peak_kw'],                  72.2,  0.1),
    ('Baseline seasonal plant COP',           b['cop'],                      4.93,  0.01),
    ('Pre-cooling electricity (kWh)',         p['kwh'],                      7992,  1),
    ('Pre-cooling total cost (RM)',           p['cost'],                     3492,  1),
    ('Pre-cooling peak demand (kW)',          p['peak_kw'],                  94.2,  0.1),
    ('Pre-cooling cost increase (%)',         100 * (p['cost'] / b['cost'] - 1), 9.0, 0.1),
    ('Pre-cooling energy increase (%)',       100 * (p['kwh'] / b['kwh'] - 1),   4.3, 0.1),
    ('Integrator max error (K)',              V['integrator_err_K'],         7.9e-5, 0.2e-5),
    ('AHRI full-load (kW/ton)',               V['ahri']['full_kw_ton'],      0.597, 0.001),
    ('AHRI IPLV (kW/ton)',                    V['ahri']['iplv_kw_ton'],      0.556, 0.001),
    ('Energy-gap, min over sensitivity (%)',  min(r['gap'] for r in s),      3.9,   0.1),
    ('Energy-gap, max over sensitivity (%)',  max(r['gap'] for r in s),      5.9,   0.1),
    ('Cost-gap, max over sensitivity (%)',    max(r['gap_cost'] for r in s), 14.4,  0.1),
    # ---- Phase 2 (tolerances allow for scikit-learn version differences) ----
    ('Forecast CV(RMSE) 15 min, GB (%)',      P2['forecast']['15']['noisy']['GradientBoosting']['cv_rmse'], 2.3, 0.4),
    ('Forecast CV(RMSE) 1 h, GB (%)',         P2['forecast']['60']['noisy']['GradientBoosting']['cv_rmse'], 3.3, 0.5),
    ('Forecast CV(RMSE) 4 h, GB (%)',         P2['forecast']['240']['noisy']['GradientBoosting']['cv_rmse'], 5.3, 0.8),
    ('Seasonal-naive CV(RMSE) 1 h (%)',       P2['forecast']['60']['noisy']['SeasonalNaive']['cv_rmse'], 10.0, 0.5),
    ('RLS power error, scalar (%)',           P2['online_scalar']['0.998']['pred_rmse_online_pct'], 2.76, 0.1),
    ('Static-model power error (%)',          P2['online_scalar']['0.998']['pred_rmse_static_pct'], 5.45, 0.1),
    ('RLS power error, curve (%)',            P2['online_curve']['pred_rmse_curve_pct'], 2.57, 0.1),
    ('Staging headroom, optimal vs rule (%)', P2['headroom_nominal']['saving_pct'], 1.43, 0.05),
    ('Rule-based loss under part-load drift (%)', P2['decision_part_load']['rule']['excess_pct'], 3.32, 0.1),
    ('Loss, forecast 1 h staging (%)',        P2['forecast_decision_measured']['60']['excess_pct'], 0.06, 0.03),
    ('Loss, model part-load error a=0.15 (%)', P2['model_error_sweep']['a']['0.15']['excess_pct'], 2.83, 0.1),
    ('Loss, model scale error, worst of sweep (%)', max(v['excess_pct'] for v in P2['model_error_sweep']['k'].values()), 0.010, 0.002),
    # ---- staging diagnostics quoted in the paper ----
    ('Rule runs too MANY chillers (% of steps)',   P2['headroom_nominal']['rule_too_many_pct'], 35.9, 0.3),
    ('Rule runs too FEW chillers (% of steps)',    P2['headroom_nominal']['rule_too_few_pct'],   0.0, 1e-9),
    ('Optimum is the minimum feasible stage (%)',  P2['headroom_nominal']['opt_is_min_feasible_pct'], 98.0, 0.3),
    ('Loss carried by the worst 10 % of steps (%)',P2['headroom_nominal']['loss_share_worst_decile_pct'], 55.2, 0.5),
    ('Median waste of a mismatched step (%)',      P2['headroom_nominal']['median_mismatch_waste_pct'], 2.80, 0.05),
    ('Stage changes, rule-based sequencer',        P2['headroom_nominal']['switches_rule'], 377, 0),
    ('Stage changes, quasi-static optimum',        P2['headroom_nominal']['switches_opt'],  532, 0),
    ('a=0.38 model picks min feasible (% of steps)', P2['model_error_sweep']['a']['0.38']['picks_min_feasible_pct'], 100.0, 1e-9),
]
bad = 0
for name, val, exp, tol in checks:
    ok = abs(val - exp) <= tol
    bad += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name:42s} got {val:10.4g}   report {exp:10.4g}")
print('\nAll reported values reproduced.' if not bad else f'\n{bad} value(s) differ from the report.')
sys.exit(1 if bad else 0)
