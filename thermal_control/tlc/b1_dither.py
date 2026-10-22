"""B1 Dither tones."""
import numpy as np

from params import ControllerParams
from .common import Mode, N


class B1Dither:
    def __init__(self, cp: ControllerParams):
        self.cp = cp
        self.f = np.array(cp.tone_freqs)
        self.w = 2 * np.pi * self.f
        self.amp_ghz = np.full(N, cp.dither_nominal)     # scheduled peak detuning
        self.cmd_amp = np.zeros(N)                       # peak command (fraction of FS)

    def schedule(self, sigma_delta, modes, fixed=None):
        """Amplitude between 0.1 and 0.5 GHz from the estimator's uncertainty (proposal B1)."""
        cp = self.cp
        if fixed is not None:
            a = np.full(N, fixed)
        else:
            a_new = np.clip(0.3 + 0.5 * sigma_delta, cp.dither_min, cp.dither_max)
            # slow schedule: amplitude changes inside a lock-in window would smear the tone bins
            a = self.amp_ghz + 0.01 * (a_new - self.amp_ghz)
            a = np.where((modes == Mode.ACQUIRE) | (modes == Mode.REACQUIRE), cp.dither_max, a)
        self.amp_ghz = a

    def set_command_amplitude(self, h_own_abs, theta):
        self.cmd_amp = self.amp_ghz / (h_own_abs * theta)

    def samples(self, t):
        """Dither commands at output sample times t (n,) -> (n, 4)."""
        return self.cmd_amp[None, :] * np.cos(self.w[None, :] * t[:, None])
