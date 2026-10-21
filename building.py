"""3R2C lumped-parameter building: air node + structural mass node.
x = [T_air, T_mass]; u = [T_out, Q_air_gain, Q_mass_gain, Q_cool_sensible]  (kW, K, h)."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

DEFAULTS = dict(Gao=8.0, Gam=30.0, Gmo=6.0,          # conductances, kW/K
                q_int_occ=95.0, q_int_base=18.0,     # internal gains, kW (occupied / unoccupied)
                q_sol_air=25.0, q_sol_mass=25.0,      # solar gains at 950 W/m2, kW
                latent=1.3, area=5000.0)


def _caps_for_taus(Gao, Gam, Gmo, tau_fast=0.85, tau_slow=10.4):
    def f(c):
        Ca, Cm = np.abs(c)
        A = np.array([[-(Gao + Gam) / Ca, Gam / Ca], [Gam / Cm, -(Gam + Gmo) / Cm]])
        lam = np.sort(np.linalg.eigvals(A).real)  # most negative first
        return [-1 / lam[0] - tau_fast, -1 / lam[1] - tau_slow]
    sol = least_squares(f, [60.0, 140.0], bounds=([5.0, 5.0], [1000.0, 3000.0]), xtol=1e-12, ftol=1e-12)
    return sol.x[0], sol.x[1]


class Building:
    def __init__(self, **kw):
        p = dict(DEFAULTS); scale = {}
        for k in ('Ca', 'Cm'):
            if k in kw: scale[k] = kw.pop(k)
        p.update(kw)
        self.__dict__.update(p)
        Ca0, Cm0 = _caps_for_taus(DEFAULTS['Gao'], DEFAULTS['Gam'], DEFAULTS['Gmo'])
        self.Ca = scale.get('Ca', Ca0); self.Cm = scale.get('Cm', Cm0)
        self._build()

    def _build(self):
        Ca, Cm, Gao, Gam, Gmo = self.Ca, self.Cm, self.Gao, self.Gam, self.Gmo
        self.A = np.array([[-(Gao + Gam) / Ca, Gam / Ca], [Gam / Cm, -(Gam + Gmo) / Cm]])
        self.B = np.array([[Gao / Ca, 1 / Ca, 0, -1 / Ca], [Gmo / Cm, 0, 1 / Cm, 0]])

    def state_space(self):
        """Continuous-time matrices exported for the Phase-3 predictive controller."""
        return self.A.copy(), self.B.copy()

    def time_constants(self):
        lam = np.sort(np.linalg.eigvals(self.A).real)
        return -1 / lam[0], -1 / lam[1]

    def gains(self, G, occ):
        qa = self.q_int_base + (self.q_int_occ - self.q_int_base) * occ + self.q_sol_air * G / 950.0
        qm = self.q_sol_mass * G / 950.0
        return qa, qm

    def deriv(self, x, u):
        return self.A @ x + self.B @ u

    def step_rk4(self, x, u, dt):
        k1 = self.deriv(x, u); k2 = self.deriv(x + 0.5 * dt * k1, u)
        k3 = self.deriv(x + 0.5 * dt * k2, u); k4 = self.deriv(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def step_exact(self, x, u, dt):
        """Closed-form solution for inputs held constant over the step (matrix exponential)."""
        M = np.zeros((3, 3)); M[:2, :2] = self.A; M[:2, 2] = self.B @ u
        return (expm(M * dt) @ np.append(x, 1.0))[:2]
