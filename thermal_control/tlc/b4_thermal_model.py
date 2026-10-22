"""Reduced-order thermal model (inside B4, also used by B7)."""
import numpy as np

from .common import N


class ThermalModel:
    def __init__(self, ts, tau_hs, tau_x):
        self.a_hs = np.exp(-ts / tau_hs)
        self.a_x = np.exp(-ts / tau_x)
        self.z_hs = np.zeros(N)
        self.z_x = np.zeros(N)
        self.gf = self.ghs = None
        self.gx = None

    def set_gains(self, gf, ghs, gx):
        self.gf, self.ghs, self.gx = gf, ghs, gx

    def settle(self, u):
        self.z_hs = self.ghs * u
        self.z_x = self.gx @ u

    def step(self, u_mean, u_end):
        self.z_hs = self.a_hs * self.z_hs + (1 - self.a_hs) * self.ghs * u_mean
        self.z_x = self.a_x * self.z_x + (1 - self.a_x) * (self.gx @ u_mean)
        return self.h(u_end)

    def h(self, u_end):
        return self.gf * u_end + self.z_hs + self.z_x
