"""B6 Setpoints."""
import numpy as np

from optics import er_db, oma_dbm
from params import ControllerParams
from .b10_calibration import B10Calibration
from .common import N


class B6Setpoints:
    """Zero-curvature (max-slope) operating point on the long-wavelength side, shifted inside
    W-OP to balance OMA across channels using an offline map (proposal B6)."""

    def __init__(self, cal: B10Calibration, cp: ControllerParams):
        self.base = cal.targets_raw.copy()
        grid = np.linspace(-cp.oma_shift_max, cp.oma_shift_max, 81)
        oma = np.array([[oma_dbm(b + s, h, d, cal.mod, p) for s in grid]
                        for b, h, d, p in zip(self.base, cal.hwhm, cal.depth, cal.p_laser)])
        # ER must stay >= er_min over +/- er_check_window around the shifted target (R13)
        w = cp.er_check_window
        er_ok = np.array([[min(er_db(b + s + e, h, d, cal.mod) for e in (-w, 0, w)) >= cp.er_min_db
                           for s in grid] for b, h, d in zip(self.base, cal.hwhm, cal.depth)])
        oma_masked = np.where(er_ok, oma, np.nan)
        best = None
        for target in np.linspace(np.nanmin(oma_masked), np.nanmax(oma_masked), 200):
            idx = np.nanargmin(np.abs(oma_masked - target), axis=1)
            vals = oma[np.arange(N), idx]
            spread = vals.max() - vals.min()
            cost = spread + 0.02 * np.abs(grid[idx]).sum()
            if best is None or cost < best[0]:
                best = (cost, idx, vals)
        self.shift = grid[best[1]]
        self.targets = self.base + self.shift
        self.oma = best[2]
        self.oma_unshifted = oma[:, len(grid) // 2]
