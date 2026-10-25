"""Conventional baselines: scheduled PI thermostat and fixed-schedule rule-based pre-cooling."""
import numpy as np


class PI:
    # umax is the PI output limit in kW SENSIBLE; the coil sees umax * Building.latent (1.3).
    # It is set to 400 kW (520 kW at the coil) so that it exceeds what the plant can deliver and
    # therefore never binds. With the earlier value of 340 kW the pre-cooling strategy saturated the
    # controller at four steps, and the resulting demand peak -- and with it the headline cost gap --
    # was set by this constant rather than by the plant: the gap is +2.1 % at umax=300, +7.3 % at 340
    # and +9.0 % at 400 and above, while the energy gap holds at 4.13-4.27 % throughout.
    # The baseline never saturates at any value. See README, "Limitations".
    def __init__(self, Kp=100.0, Ti=1.5, umax=400.0, Tt=0.5):
        self.Kp, self.Ti, self.umax, self.Tt = Kp, Ti, umax, Tt
        self.I = 0.0

    def reset(self):
        self.I = 0.0

    def step(self, Ta, sp, dt):
        e = Ta - sp
        u_raw = self.Kp * e + self.I
        u = min(max(u_raw, 0.0), self.umax)
        # integral with back-calculation anti-windup
        self.I += dt * (self.Kp / self.Ti * e + (u - u_raw) / self.Tt)
        return u


class Baseline:
    name = 'Baseline: schedule + PI'
    def __init__(self):
        self.pi = PI()
    def schedule(self, h):
        return (6.0 <= h < 18.5), 23.5
    def command(self, Ta, h, dt):
        on, sp = self.schedule(h)
        if not on:
            self.pi.reset(); return 0.0
        return self.pi.step(Ta, sp, dt)


class PreCool(Baseline):
    name = 'Rule-based pre-cooling'
    def schedule(self, h):
        if 4.5 <= h < 14.0: return True, 22.5
        if 14.0 <= h < 18.5: return True, 24.0
        return False, 23.5


def make(strategy):
    return {'baseline': Baseline, 'precool': PreCool}[strategy]()
