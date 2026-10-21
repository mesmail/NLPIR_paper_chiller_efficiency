"""Online efficiency identification: scalar recursive least squares with forgetting (gray-box).

Measured chiller electrical power is modelled as  P_meas = k * P_static(q, n, T_cw),
where P_static is the commissioning (DOE-2 EIR) model and k is a slowly varying efficiency multiplier
(k = 1 / relative COP).  k is estimated recursively from sub-metered power, load, stage count and condenser
water temperature."""
import numpy as np
from chiller import Plant


class ScalarRLS:
    def __init__(self, lam=0.998, theta0=1.0, p0=1.0):
        self.lam, self.theta, self.P = lam, theta0, p0

    def update(self, phi, y):
        K = self.P * phi / (self.lam + phi * self.P * phi)
        err = y - self.theta * phi
        self.theta += K * err
        self.P = (self.P - K * phi * self.P) / self.lam
        return err


def static_chiller_power(plant, q, n, t_cw):
    p, _ = plant.power_unit_total(q, n, t_cw)
    return p


def identify(R, noise=0.02, lam=0.998, seed=11, plant=None):
    """Run the estimator over a simulated record R. Returns per-step arrays aligned with R (nan when plant off)."""
    plant = plant or Plant()
    rng = np.random.default_rng(seed)
    N = len(R['t']); k_hat = np.full(N, np.nan); pred_err = np.full(N, np.nan)
    pred_static_err = np.full(N, np.nan)
    est = ScalarRLS(lam=lam); last = 1.0
    for i in range(N):
        if R['n'][i] > 0 and R['q_coil'][i] > 0.5:
            phi = static_chiller_power(plant, R['q_coil'][i], int(R['n'][i]), R['t_cw'][i])
            y = R['P_ch'][i] * (1.0 + noise * rng.standard_normal())
            pred_err[i] = (est.theta * phi - y) / y             # one-step-ahead, before the update
            pred_static_err[i] = (1.0 * phi - y) / y
            est.update(phi, y); last = est.theta
        k_hat[i] = last
    return dict(k_hat=k_hat, pred_err=pred_err, pred_static_err=pred_static_err)


def fouling_cleaning(t_hours):
    """Injected drift: COP multiplier m(t) (< 1 means worse). Fouling from day 15 to 60 (-10 %), cleaning at day 60
    (back to -1 %), then slower fouling (-6 % over 45 days)."""
    d = t_hours / 24.0
    if d < 60:
        return 1.0 - 0.10 * float(np.clip((d - 15) / 45.0, 0, 1))
    return 0.99 - 0.06 * float(np.clip((d - 60) / 45.0, 0, 1))


def part_load_drift(t_hours):
    """Injected drift 2: the part-load curve EIRfPLR = a + b*PLR + c*PLR^2 flattens as the fixed-loss term a grows
    from 0.30 to 0.42 (days 20-80); b and c are rescaled so that the curve still equals 1 at full load."""
    d = t_hours / 24.0
    a = 0.30 + 0.12 * float(np.clip((d - 20) / 60.0, 0, 1))
    f = (1.0 - a) / 0.70
    return (a, 0.23 * f, 0.47 * f)


class VectorRLS:
    def __init__(self, theta0, lam=0.999, p0=0.05):
        self.theta = np.array(theta0, float); self.P = np.eye(len(theta0)) * p0; self.lam = lam

    def update(self, phi, y):
        phi = np.asarray(phi, float); Pphi = self.P @ phi
        K = Pphi / (self.lam + phi @ Pphi)
        err = y - self.theta @ phi
        self.theta = self.theta + K * err
        self.P = (self.P - np.outer(K, Pphi)) / self.lam
        return err


def identify_curve(R, noise=0.02, lam=0.999, seed=11, plant=None):
    """Three-parameter gray-box estimator of the part-load curve: P_ch = n*cap_u/COP_ref * EIRfT * (a + b*PLR + c*PLR^2)
    is linear in (a, b, c). Returns per-step estimates (carried forward when the plant is off)."""
    from chiller import biquad
    plant = plant or Plant()
    rng = np.random.default_rng(seed)
    N = len(R['t']); th = np.full((N, 3), np.nan); perr = np.full(N, np.nan)
    est = VectorRLS(plant.eirfplr, lam=lam); last = np.array(plant.eirfplr)
    for i in range(N):
        if R['n'][i] > 0 and R['q_coil'][i] > 0.5:
            n = int(R['n'][i]); tcw = R['t_cw'][i]; capu = plant.cap_unit(tcw)
            plr = min(R['q_coil'][i] / (n * capu), 1.0)
            if plr < plant.plr_min:                       # cycling region: excluded from identification
                th[i] = last; continue
            eirt = biquad(plant.eirft, plant.t_chw - 6.7, tcw - plant.t_cw_ref)
            base = n * capu / plant.cop_ref * eirt
            phi = base * np.array([1.0, plr, plr * plr])
            y = R['P_ch'][i] * (1.0 + noise * rng.standard_normal())
            perr[i] = (est.theta @ phi - y) / y
            est.update(phi, y); last = est.theta.copy()
        th[i] = last
    return dict(theta=th, pred_err=perr)
