"""B8 Output stage."""
import numpy as np

from params import ControllerParams
from .b1_dither import B1Dither
from .common import N


class B8Output:
    def __init__(self, cp: ControllerParams):
        self.cp = cp
        self.n_sub = int(round(cp.ts * cp.out_rate))
        self.dt = 1.0 / cp.out_rate
        self.slew = 0.10 / 1e-3 * self.dt          # 10 % FS per ms (R19), per output sample
        self.last = np.zeros(N)                    # last output sample (with dither)
        self.u_prev = np.zeros(N)                  # last command without dither
        self.limit_hits = 0
        self.slew_hits = 0
        self.max_step = 0.0          # largest output change per sample (R19 audit)
        self.out_min, self.out_max = 1.0, 0.0

    def produce(self, u_new, t_start, dither: B1Dither, guard_hi):
        """First-order-hold interpolation of the command + dither, clamped, slew limited."""
        u_c = np.clip(u_new, 0.0, guard_hi)
        frac = (np.arange(1, self.n_sub + 1) / self.n_sub)[:, None]
        base = self.u_prev[None, :] + frac * (u_c - self.u_prev)[None, :]
        ts = t_start + np.arange(self.n_sub) * self.dt
        dith = dither.samples(ts)
        seq = base + dith
        out = np.empty_like(seq)
        prev = self.last
        for k in range(self.n_sub):
            v = np.clip(seq[k], 0.0, 1.0)
            dv = np.clip(v - prev, -self.slew, self.slew)
            self.slew_hits += int(np.any(np.abs(v - prev) > self.slew + 1e-12))
            v = prev + dv
            self.max_step = max(self.max_step, float(np.max(np.abs(v - prev))))
            self.out_min = min(self.out_min, float(v.min()))
            self.out_max = max(self.out_max, float(v.max()))
            out[k] = v
            prev = v
        self.limit_hits += int(np.any((u_c <= 0) | (u_c >= guard_hi)))
        self.last = prev
        # what was actually applied, without the dither (the thermal model must see this, not the
        # request, or a slew-limited move would be mis-predicted on every neighbour)
        applied = out - dith
        u_mean_nodither = applied.mean(0)
        u_c = applied[-1]
        self.u_prev = u_c
        return out, u_c, u_mean_nodither
