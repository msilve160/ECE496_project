"""B2 Conditioner: FIR + decimation + validity / squelch flags."""
import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi

from params import ControllerParams
from .common import N


class B2Conditioner:
    def __init__(self, cp: ControllerParams, pd_rate):
        self.cp = cp
        self.h = firwin(cp.fir_taps, cp.fir_cutoff, fs=pd_rate)
        self.delay = (cp.fir_taps - 1) / 2 / pd_rate
        self.zi = np.zeros((cp.fir_taps - 1, N))
        self.last_valid = np.full(N, np.nan)
        self.noise = np.full(N, 1e-3)              # running rms noise estimate (FS)
        self.noise_gain = np.sqrt(np.sum(self.h ** 2))
        lsb = 1 / 4096
        self.lsb = lsb

    def process(self, x):
        """x: (n, 4) raw PD samples. Returns decimated filtered samples (n/decim, 4) and flags."""
        missing = np.isnan(x)
        sat = (x >= 1 - 2 * self.lsb) | (x <= self.lsb)
        # fill gaps with last valid value so the filter state stays sane
        xf = x.copy()
        for i in range(N):
            if missing[:, i].any():
                fill = self.last_valid[i] if np.isfinite(self.last_valid[i]) else 0.0
                col = xf[:, i]
                col[missing[:, i]] = fill
            self.last_valid[i] = xf[-1, i]
        stuck = np.std(xf, axis=0) < 0.5 * self.lsb
        invalid = missing.any(0) | (sat.mean(0) > 0.01) | stuck
        if not getattr(self, "_primed", False):
            self.zi = lfilter_zi(self.h, 1.0)[:, None] * xf[0][None, :]
            self._primed = True
        y, self.zi = lfilter(self.h, 1.0, xf, axis=0, zi=self.zi)
        # wideband noise from the second difference (removes the slow dither / signal)
        d2 = np.diff(xf, 2, axis=0)
        nz = np.std(d2, axis=0) / np.sqrt(6)
        self.noise = np.where(invalid, self.noise, 0.7 * self.noise + 0.3 * nz)
        self.noise_inst = np.where(invalid, self.noise, nz)
        return y[::self.cp.decim], invalid
