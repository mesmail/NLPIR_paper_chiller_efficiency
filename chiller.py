"""Water-cooled centrifugal plant: DOE-2 / EnergyPlus 'Electric EIR' chillers, tower, pumps, sequencing."""
import numpy as np


def biquad(c, x, y):
    return c[0] + c[1] * x + c[2] * x * x + c[3] * y + c[4] * y * y + c[5] * x * y


class Plant:
    def __init__(self, n=3, cap=150.0, cop_ref=5.9, t_chw=6.7, t_cw_ref=29.4,
                 up=0.90, down=0.75, plr_min=0.15, cop_scale=1.0):
        self.n, self.cap, self.cop_ref = n, cap, cop_ref * cop_scale
        self.cop_ref0 = self.cop_ref
        self.t_chw, self.t_cw_ref = t_chw, t_cw_ref
        self.up, self.down, self.plr_min = up, down, plr_min
        # curve coefficients are SYNTHETIC (not from a manufacturer catalogue); x = T_chw - ref, y = T_cw - ref
        self.capft = (1.0, 0.020, 0.0, -0.012, 0.0, 0.0)
        self.eirft = (1.0, -0.015, 0.0, 0.018, 0.0002, 0.0)
        self.eirfplr = (0.30, 0.23, 0.47)               # a + b*PLR + c*PLR^2, equals 1 at PLR = 1
        self.q_design = 400.0
        self.n_prev = 0

    # ---- chiller curves -------------------------------------------------
    def cap_unit(self, t_cw):
        return self.cap * biquad(self.capft, self.t_chw - 6.7, t_cw - self.t_cw_ref)

    def power_unit_total(self, q, n, t_cw):
        """Electrical input (kW) for n identical units sharing evaporator load q (kW)."""
        capu = self.cap_unit(t_cw)
        plr = q / (n * capu)
        a, b, c = self.eirfplr
        eirt = biquad(self.eirft, self.t_chw - 6.7, t_cw - self.t_cw_ref)
        def p_at(pl):
            return n * capu / self.cop_ref * eirt * (a + b * pl + c * pl * pl)
        if plr >= self.plr_min:
            return p_at(min(plr, 1.0)), plr
        return p_at(self.plr_min) * plr / self.plr_min, plr      # on/off cycling below minimum PLR

    def chiller_cop_at(self, plr, t_cw):
        a, b, c = self.eirfplr
        eirt = biquad(self.eirft, self.t_chw - 6.7, t_cw - self.t_cw_ref)
        return self.cop_ref / (eirt * (a + b * plr + c * plr * plr) / plr)

    # ---- sequencing -----------------------------------------------------
    def n_required(self, q, cap_u, n_prev):
        n_up = int(np.ceil(q / (self.up * cap_u) - 1e-9)); n_up = max(1, min(self.n, n_up))
        if n_prev == 0:
            return n_up
        if n_up > n_prev:
            return n_up
        if n_prev > 1 and q < self.down * (n_prev - 1) * cap_u:
            return n_prev - 1
        return n_prev

    # ---- whole plant ----------------------------------------------------
    def operate(self, q_req, t_wb, n_prev=None, ramp=False, n_force=None):
        if n_prev is None: n_prev = self.n_prev
        if q_req < 0.5:
            self.n_prev = 0
            return dict(q=0.0, n=0, P_ch=0.0, P_cwp=0.0, P_fan=0.0, P_chwp=0.0, t_cw=t_wb + 3.0, plr=0.0)
        t_cw = t_wb + 4.0
        for _ in range(4):
            capu = self.cap_unit(t_cw)
            n = n_force if n_force else self.n_required(q_req, capu, 0 if ramp else n_prev)
            q = min(q_req, n * capu)
            P_ch, plr = self.power_unit_total(q, n, t_cw)
            q_rej = q + P_ch
            frac = q_rej / (self.n * self.cap * 1.2)
            t_cw = t_wb + 3.0 + 2.0 * min(frac, 1.0)
        frac_u = min(1.0, q_rej / (n * self.cap * 1.2))
        P_cwp = 1.6 * n
        P_fan = n * 3.0 * (0.25 + 0.75 * frac_u ** 3)
        P_chwp = 5.0 * max(min(1.0, q / self.q_design), 0.25) ** 2.5
        self.n_prev = n
        return dict(q=q, n=n, P_ch=P_ch, P_cwp=P_cwp, P_fan=P_fan, P_chwp=P_chwp, t_cw=t_cw, plr=plr)

    @staticmethod
    def p_total(r):
        return r['P_ch'] + r['P_cwp'] + r['P_fan'] + r['P_chwp']

    def plant_cop_curve(self, loads_kw, t_wb):
        out = []
        for q in loads_kw:
            r = self.operate(q, t_wb, ramp=True)
            out.append(r['q'] / self.p_total(r) if r['n'] else np.nan)
        return np.array(out)

    # ---- AHRI 550/590 rating -------------------------------------------
    def ahri(self):
        ton = 3.517
        pts = [(1.00, 29.44, 0.01), (0.75, 23.89, 0.42), (0.50, 18.33, 0.45), (0.25, 18.33, 0.12)]
        cops = [self.chiller_cop_at(plr, tcw) for plr, tcw, _ in pts]
        iplv = 1.0 / sum(w / c for (_, _, w), c in zip(pts, cops))
        return dict(full_cop=cops[0], full_kw_ton=ton / cops[0], iplv_cop=iplv, iplv_kw_ton=ton / iplv, cops=cops, pts=pts)
