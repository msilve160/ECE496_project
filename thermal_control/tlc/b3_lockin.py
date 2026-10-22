"""B3 Digital lock-in demodulator (fundamental and 2nd harmonic, all tones)."""
from collections import deque

import numpy as np

from params import ControllerParams
from .common import N


class B3LockIn:
    def __init__(self, cp: ControllerParams, out_delay, fs_dec):
        self.cp = cp
        self.nwin = int(round(cp.lockin_window * fs_dec))
        self.buf = deque(maxlen=self.nwin)
        self.tbuf = deque(maxlen=self.nwin)
        self.vbuf = deque(maxlen=int(round(cp.lockin_window / cp.ts)))
        self.w = 2 * np.pi * np.array(cp.tone_freqs)
        self.out_delay = out_delay

    def push(self, y, t, invalid):
        for k in range(y.shape[0]):
            self.buf.append(y[k])
            self.tbuf.append(t[k])
        self.n_pushed = getattr(self, "n_pushed", 0) + y.shape[0]
        self.vbuf.append(invalid)

    def demod(self, h_complex):
        """Returns y0 (4,), own-tone y1, y2 (4,), cross-tone projections (4,4) and validity."""
        x = np.array(self.buf)                       # (n, 4)
        t = np.array(self.tbuf) - self.out_delay
        ph = self.w[None, :] * t[:, None]            # (n, 4 tones)
        c1, s1 = np.cos(ph), np.sin(ph)
        c2, s2 = np.cos(2 * ph), np.sin(2 * ph)
        # Hann window: suppresses leakage of slow detuning trends into the tone bins
        n = x.shape[0]
        if getattr(self, "_win", None) is None or len(self._win) != n:
            self._win = np.hanning(n + 2)[1:-1]
            self._win /= self._win.sum()
        xw = x * self._win[:, None]
        y0 = xw.sum(0)
        i1 = 2 * xw.T @ c1                           # [ring, tone]
        q1 = 2 * xw.T @ s1
        i2 = 2 * xw.T @ c2
        q2 = 2 * xw.T @ s2
        # project on the expected phase: wobble ~ |H| cos(w t + arg H)
        phi = np.angle(h_complex)                    # [ring, tone]
        y1_all = i1 * np.cos(phi) - q1 * np.sin(phi)
        own = np.arange(N)
        y1 = y1_all[own, own]
        p2 = 2 * phi[own, own]
        y2 = i2[own, own] * np.cos(p2) - q2[own, own] * np.sin(p2)
        self.n_pushed = getattr(self, "n_pushed", 0)
        valid = ~np.any(np.array(self.vbuf), axis=0) & (self.n_pushed >= int(1.5 * self.nwin))
        valid &= y0 > 0.01
        self.own_iq = i1[own, own] - 1j * q1[own, own]   # = A exp(j phase) of the own tone
        return y0, y1, y2, y1_all, valid
