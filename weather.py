"""Tropical ambient conditions and occupancy schedule (5-minute base grid)."""
import numpy as np

BASE_DT_H = 1.0 / 12.0


def twb_stull(T, RH):
    """Wet-bulb temperature (deg C) from dry-bulb (deg C) and RH (%) - Stull (2011)."""
    return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + np.arctan(T + RH)
            - np.arctan(RH - 1.676331) + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
            - 4.686035)


def occupancy(h):
    """Fraction of design occupancy vs hour of day (weekday office)."""
    h = np.asarray(h, dtype=float)
    ramp_up = np.clip((h - 7.5) / 1.0, 0, 1)
    ramp_dn = np.clip((18.5 - h) / 1.0, 0, 1)
    occ = np.minimum(ramp_up, ramp_dn)
    lunch = (h >= 12.0) & (h < 13.0)
    occ = np.where(lunch, occ * 0.8, occ)
    return occ


def make_weather(days=16, seed=7):
    rng = np.random.default_rng(seed)
    n = int(round(days * 24 / BASE_DT_H))
    t = np.arange(n) * BASE_DT_H
    h = t % 24.0
    d = (t // 24).astype(int)
    off = np.zeros(days); cloud = np.zeros(days)
    x = 0.0; cl = 0.75
    for i in range(days):
        x = 0.6 * x + rng.normal(0, 0.8); off[i] = x
        cl = float(np.clip(0.6 * cl + 0.4 * 0.75 + rng.normal(0, 0.18), 0.35, 1.0)); cloud[i] = cl
    storm_day = rng.random(days) < 0.45
    s_start = rng.uniform(14.5, 16.5, days); s_len = rng.uniform(1.0, 2.0, days)
    st = np.zeros(n)
    for i in range(days):
        if storm_day[i]:
            m = (d == i) & (h >= s_start[i]) & (h < s_start[i] + s_len[i])
            st[m] = np.sin(np.pi * (h[m] - s_start[i]) / s_len[i])
    # smooth AR(1) noise on dry-bulb
    e = rng.normal(0, 0.12, n); nz = np.zeros(n)
    for k in range(1, n):
        nz[k] = 0.97 * nz[k - 1] + e[k]
    Tdb = 28.0 + off[d] + 4.2 * np.sin(2 * np.pi * (h - 9) / 24) + nz - 4.5 * st
    G = np.maximum(0, np.sin(np.pi * (h - 6.5) / 12)) * 950.0 * cloud[d] * (1 - 0.75 * st)
    RH = np.clip(74 - 12 * np.sin(2 * np.pi * (h - 9) / 24) + 10 * st, 50, 98)
    return dict(t=t, h=h, d=d, Tdb=Tdb, RH=RH, G=G, occ=occupancy(h))


def resample(w, dt_h):
    """Block-average the 5-min weather onto a coarser control step."""
    f = int(round(dt_h / BASE_DT_H))
    n = (len(w['t']) // f) * f
    out = {}
    for k in ('Tdb', 'RH', 'G', 'occ'):
        out[k] = w[k][:n].reshape(-1, f).mean(axis=1)
    out['t'] = w['t'][:n:f]
    out['h'] = out['t'] % 24.0
    out['d'] = (out['t'] // 24).astype(int)
    out['Twb'] = twb_stull(out['Tdb'], out['RH'])
    return out
