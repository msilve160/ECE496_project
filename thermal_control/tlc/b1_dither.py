"""B1 Dither tones."""
import numpy as np

from params import ControllerParams
from .common import Mode, N


K_HARM = (1, 3, 5, 7, 9)       # harmonics of a square dither that the estimator models


class B1Dither:
    def __init__(self, cp: ControllerParams):
        self.cp = cp
        self.f = np.array(cp.tone_freqs)
        self.w = 2 * np.pi * self.f
        self.square = cp.dither_shape != "sine"      # True for every sampled-pattern shape
        # fundamental amplitude of a unit-peak waveform: 1 (sine), 4/pi (square)
        self.fund = np.full(N, 4 / np.pi) if self.square else np.ones(N)
        # R19: dither edges may use 40 % of the slew budget (the rest is left to the base command),
        # so a square wave of large amplitude has ramped edges (trapezoid)
        self.edge_step = 0.35 * 0.10 / 1e-3 / cp.out_rate
        self.b = np.zeros((len(K_HARM), N), complex)     # harmonic coefficients of the actual wave
        self._built = np.zeros(N)
        self.phi1 = np.zeros(N)                          # phase of the fundamental (ramp delay), rad
        if self.square:
            self.n_per = np.rint(cp.out_rate / self.f).astype(int)
            self.pat = [np.zeros(n) for n in self.n_per]
            self._rebuild(np.ones(N) * 1e-3, force=True)
        self.amp_ghz = np.full(N, cp.dither_nominal)     # scheduled fundamental detuning amplitude
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
        self.cmd_amp = self.amp_ghz / (self.fund * h_own_abs * theta)
        if self.square:
            # fixed wave shape for the largest scheduled amplitude: the edges then never exceed the
            # slew budget at smaller amplitudes, and the harmonic content only rescales
            self._rebuild(self.cp.dither_max / (0.7 * h_own_abs * theta))   # fund assumed >= 0.7, no feedback

    def _rebuild(self, cmd_amp, force=False):
        """Periodic one-period pattern (unit peak) per tone: sign(cos) with edges slew-limited to
        edge_step, and its harmonic coefficients b[k] (wave = Re sum_k cmd * b_k e^{jk w (t - Ts/2)},
        b_1 real positive, so the delay of the ramps is absorbed in the phase of the harmonics)."""
        for i in range(N):
            c = max(cmd_amp[i], 1e-9)
            if not force and abs(c - self._built[i]) < 0.10 * self._built[i]:
                continue
            self._built[i] = c
            n = self.n_per[i]
            cs = np.cos(2 * np.pi * np.arange(n) / n + 1e-6)
            shape = self.cp.dither_shape
            if shape == "square":
                tgt = np.where(cs >= 0, 1.0, -1.0)
            elif shape == "triangle":                    # constant slope: best use of the slew budget
                ph = (np.arange(n) / n + 0.5) % 1.0 - 0.5
                tgt = 1.0 - 4.0 * np.abs(ph)
            elif shape == "three":                       # +1 / 0 / -1: no 3rd harmonic
                tgt = np.where(cs > 0.5, 1.0, np.where(cs < -0.5, -1.0, 0.0))
            else:                                        # "qsine": sine rounded to 5 levels
                tgt = np.round(2 * cs) / 2
            lim = self.edge_step / c
            x, v = np.empty(n), tgt[-1]
            for rep in range(2):
                for m in range(n):
                    v = v + np.clip(tgt[m] - v, -lim, lim)
                    x[m] = v
            self.pat[i] = x
            for r, k in enumerate(K_HARM):
                th = 2 * np.pi * k / n
                z = (2.0 / n) * np.sum(x * np.exp(-1j * th * np.arange(n))) * np.sinc(th / 2 / np.pi)
                self.b[r, i] = z
            b1 = self.b[0, i]
            self.phi1[i] = np.angle(b1)
            for r, k in enumerate(K_HARM):
                self.b[r, i] *= np.exp(-1j * k * np.angle(b1))
            self.fund[i] = abs(b1)

    def samples(self, t):
        """Dither commands at output sample times t (n,) -> (n, 4). The square wave is what an NCO
        phase accumulator plus a sign bit gives on an FPGA: sign(cos) at the 10 us sample instants."""
        if self.square:
            idx = np.rint(t * self.cp.out_rate).astype(np.int64)
            wave = np.stack([self.pat[i][idx % self.n_per[i]] for i in range(N)], axis=1)
        else:
            wave = np.cos(self.w[None, :] * t[:, None])
        return self.cmd_amp[None, :] * wave
