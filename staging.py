"""Quasi-static staging analysis: total plant power as a function of the number of running chillers."""
import numpy as np
from chiller import Plant


def power_for_n(plant, q, twb, n, k=1.0):
    """Total plant electrical power (kW) with n chillers running, scaling chiller power by k.
    Returns np.inf if n chillers cannot deliver q (kW)."""
    if q < 0.5:
        return 0.0
    r = plant.operate(q, twb, n_force=n)
    if r['q'] < q - 1e-6:
        return np.inf
    return k * r['P_ch'] + r['P_cwp'] + r['P_fan'] + r['P_chwp']


def all_n(plant, q, twb, k=1.0):
    return np.array([power_for_n(plant, q, twb, n, k) for n in range(1, plant.n + 1)])


def optimal_n(plant, q, twb, k=1.0):
    p = all_n(plant, q, twb, k)
    return int(np.argmin(p)) + 1


def min_feasible_n(plant, q, twb):
    p = all_n(plant, q, twb)
    return int(np.argmax(np.isfinite(p))) + 1
