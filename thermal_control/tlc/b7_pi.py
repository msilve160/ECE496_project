"""B7 PI fallback (also the baseline controller)."""
import numpy as np

from params import ControllerParams
from .common import N


class PIController:
    """Per-ring PI on the lock-in slope error with back-calculation anti-windup.
    Used as the TLC fallback (B7) and as the baseline controller (App. C.3)."""

    def __init__(self, cp: ControllerParams, dc_gain):
        self.cp = cp
        self.g = dc_gain                     # GHz per unit u (for gain normalisation)
        self.integ = np.zeros(N)
        self.u = np.zeros(N)

    def reset(self, u):
        self.integ = np.asarray(u, float).copy()
        self.u = self.integ.copy()

    def step(self, err, valid, hi):
        cp = self.cp
        kp = cp.pi_kp / self.g
        ki = cp.pi_ki / self.g
        e = np.where(valid, err, 0.0)
        u_un = self.integ - kp * e
        u_sat = np.clip(u_un, 0.0, hi)
        self.integ += cp.ts * (-ki * e) + (u_sat - u_un) * cp.ts / cp.pi_tt
        self.integ = np.where(valid, self.integ, self.u)     # freeze while invalid
        self.u = np.where(valid, u_sat, self.u)
        return self.u.copy()
