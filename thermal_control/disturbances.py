"""Design-basis disturbances D1-D6 (proposal Table II) plus fault injections.

Each component is evaluated on a vector of sample times and adds into a dict of signals:
  dT      (n, 4) K     temperature added at each ring
  laser   (n, 4) GHz   laser wavelength shift (positive = longer wavelength), lowers detuning
  pscale  (n, 4)       multiplicative laser power factor
  squelch (n, 4) bool  modulation squelched (D6)
  pdfault (n, 4) int   0 ok, 1 missing (NaN), 2 stuck, 3 saturated
"""
import numpy as np

N = 4


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


class D1Drift:
    """Slow common temperature drift: piecewise-linear profile of (time, K) points."""

    def __init__(self, points):
        self.tp, self.vp = np.array(points, dtype=float).T

    def apply(self, t, s):
        s["dT"] += np.interp(t, self.tp, self.vp)[:, None]


class D2Aggressor:
    """Fast on-die aggressor at the rings.

    Assumed ring-level form (see report): a ~130 ms periodic swing that keeps the published peak
    rates (+2200 / -2300 K/s) but is limited to `amp` K at the rings (3 K, from R3's rationale).
    Edges are raised-cosine, so the peak rate is reached mid-edge. `ring=None` applies it to all
    rings equally; otherwise it is local beside `ring` with a spatial decay `profile`.
    """

    def __init__(self, t0=0.0, amp=3.0, period=0.130, rise_rate=2200.0, fall_rate=2300.0,
                 n_periods=2, ring=None, profile=(1.0, 0.35, 0.12, 0.04), high_frac=0.5):
        self.t0, self.amp, self.period, self.n = t0, amp, period, n_periods
        self.tr = np.pi * amp / (2 * rise_rate)
        self.tf = np.pi * amp / (2 * fall_rate)
        self.high = high_frac * period
        if ring is None:
            self.w = np.ones(N)
        else:
            self.w = np.array([profile[abs(i - ring)] for i in range(N)])

    def waveform(self, t):
        tt = t - self.t0
        active = (tt >= 0) & (tt < self.n * self.period)
        tp = np.mod(np.maximum(tt, 0), self.period)
        rise = _smoothstep(tp / self.tr)
        fall = 1 - _smoothstep((tp - self.high) / self.tf)
        v = np.where(tp < self.high, rise, fall)
        return np.where(active, self.amp * v, 0.0)

    def apply(self, t, s):
        s["dT"] += self.waveform(t)[:, None] * self.w[None, :]

    def end_time(self):
        """Time the last falling edge has finished."""
        return self.t0 + (self.n - 1) * self.period + self.high + self.tf


class D3Uneven:
    """Uneven heating: per-ring temperature offsets (K) faded in over t_ramp."""

    def __init__(self, offsets, t0=0.0, t_ramp=0.05):
        self.off = np.asarray(offsets, float)
        self.t0, self.tr = t0, t_ramp

    def apply(self, t, s):
        s["dT"] += _smoothstep((t - self.t0) / self.tr)[:, None] * self.off[None, :]


class D4Laser:
    """Laser wavelength: fixed initial offset per ring plus a linear drift over [t0, t0+dur]."""

    def __init__(self, offset_ghz=(0, 0, 0, 0), drift_ghz=(0, 0, 0, 0), t0=0.0, dur=1.0):
        self.off = np.asarray(offset_ghz, float)
        self.drift = np.asarray(drift_ghz, float)
        self.t0, self.dur = t0, dur

    def apply(self, t, s):
        frac = np.clip((t - self.t0) / self.dur, 0, 1)
        s["laser"] += self.off[None, :] + frac[:, None] * self.drift[None, :]


class D5Power:
    """Laser power change: linear from 1 to (1 + change) over [t0, t0+dur] (all channels)."""

    def __init__(self, change=0.025, t0=0.0, dur=1.0, rings=None):
        self.change, self.t0, self.dur = change, t0, dur
        self.mask = np.ones(N) if rings is None else np.isin(np.arange(N), rings).astype(float)

    def apply(self, t, s):
        frac = np.clip((t - self.t0) / self.dur, 0, 1)
        s["pscale"] *= 1 + self.change * frac[:, None] * self.mask[None, :]


class D6Squelch:
    def __init__(self, t0, dur=0.075, rings=(0, 1, 2, 3)):
        self.t0, self.dur, self.rings = t0, dur, list(rings)

    def apply(self, t, s):
        on = (t >= self.t0) & (t < self.t0 + self.dur)
        for r in self.rings:
            s["squelch"][:, r] |= on


class TempKick:
    """Abrupt local temperature step (used to force loss of lock, R10)."""

    def __init__(self, ring, t0, kelvin, t_rise=20e-6, t_off=None):
        self.ring, self.t0, self.k, self.tr, self.toff = ring, t0, kelvin, t_rise, t_off

    def apply(self, t, s):
        v = self.k * _smoothstep((t - self.t0) / self.tr)
        if self.toff is not None:
            v = v * (1 - _smoothstep((t - self.toff) / self.tr))
        s["dT"][:, self.ring] += v


class PDFault:
    def __init__(self, ring, t0, dur, kind=1):
        self.ring, self.t0, self.dur, self.kind = ring, t0, dur, kind

    def apply(self, t, s):
        on = (t >= self.t0) & (t < self.t0 + self.dur)
        s["pdfault"][on, self.ring] = self.kind


class Disturbances:
    def __init__(self, components=()):
        self.components = list(components)

    def evaluate(self, t):
        n = len(t)
        s = {
            "dT": np.zeros((n, N)),
            "laser": np.zeros((n, N)),
            "pscale": np.ones((n, N)),
            "squelch": np.zeros((n, N), bool),
            "pdfault": np.zeros((n, N), int),
        }
        for c in self.components:
            c.apply(t, s)
        return s
