"""Time-of-use energy charge plus maximum-demand charge (PLACEHOLDER rates, RM)."""
import numpy as np
PEAK_START, PEAK_END = 14.0, 22.0
RATE_PEAK, RATE_OFFPEAK = 0.365, 0.224      # RM/kWh
MD_CHARGE = 30.30                            # RM/kW/month, pro-rated to the run length


def rate(h):
    h = np.asarray(h)
    return np.where((h >= PEAK_START) & (h < PEAK_END), RATE_PEAK, RATE_OFFPEAK)


def cost(energy_kwh_steps, h_steps, peak_kw, days):
    e = float(np.sum(energy_kwh_steps * rate(h_steps)))
    md = MD_CHARGE * peak_kw * days / 30.0
    return e + md, e, md
