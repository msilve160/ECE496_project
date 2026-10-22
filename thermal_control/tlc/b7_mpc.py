"""B7 Linear MPC solved by the fast gradient method."""
import numpy as np

from params import ControllerParams
from .b4_thermal_model import ThermalModel
from .common import N


class B7MPC:
    def __init__(self, cp: ControllerParams):
        self.cp = cp
        self.nh = cp.horizon
        self.nv = N * self.nh
        self.U = None
        self.iters_used = 0
        self._cache = {}

    def build(self, model: ThermalModel):
        """Condensed prediction matrices: h_pred = free(z) + G U  (U ordered step-major)."""
        nh = self.nh
        G = np.zeros((N * nh, N * nh))
        for j in range(nh):          # prediction step j+1
            for i in range(j + 1):   # input applied at step i+1
                blk = np.zeros((N, N))
                if i == j:
                    blk += np.diag(model.gf)
                blk += (model.a_hs ** (j - i)) * (1 - model.a_hs) * np.diag(model.ghs)
                blk += (model.a_x ** (j - i)) * (1 - model.a_x) * model.gx
                G[j * N:(j + 1) * N, i * N:(i + 1) * N] = blk
        self.G = G
        self.pw_hs = model.a_hs ** np.arange(1, nh + 1)
        self.pw_x = model.a_x ** np.arange(1, nh + 1)
        D = np.eye(N * nh) - np.eye(N * nh, k=-N)
        self.D = D
        self.DtD = D.T @ D
        self.model = model
        self._cache = {}

    def solve(self, delta_hat, rate_hat, h_now, u_prev, target, qmask, lo, hi):
        cp = self.cp
        m = self.model
        nh = self.nh
        # free response of the thermal model + estimated drift extrapolation (feedforward)
        free = (self.pw_hs[:, None] * m.z_hs[None, :] + self.pw_x[:, None] * m.z_x[None, :]).ravel()
        decay = cp.rate_decay ** np.arange(1, nh + 1)
        drift = (np.cumsum(decay)[:, None] * rate_hat[None, :] * cp.ts).ravel()
        c = np.tile(delta_hat - h_now, nh) + free + drift
        tgt = np.tile(target, nh)
        qv = np.tile(qmask * cp.q_weight, nh)
        G = self.G
        key = tuple(qmask)
        if key not in self._cache:
            M = G.T @ (qv[:, None] * G) + cp.r_weight * np.eye(self.nv) + cp.s_weight * self.DtD
            ev = np.linalg.eigvalsh(M)
            L = 2 * (ev[-1] + cp.slew_penalty * 4)
            # strong convexity over the free variables (fixed rings are pinned by their bounds)
            free = np.tile(qmask > 0, nh)
            mu = 2 * (np.linalg.eigvalsh(M[np.ix_(free, free)])[0] if free.any() else ev[0])
            beta = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
            self._cache[key] = (M, L, beta)
        M, L, beta = self._cache[key]
        b = np.zeros(self.nv); b[:N] = u_prev
        q = G.T @ (qv * (c - tgt)) - cp.s_weight * (self.D.T @ b)
        lo_v, hi_v = np.tile(lo, nh), np.tile(hi, nh)
        if self.U is None:
            U = np.tile(u_prev, nh)
        else:
            U = np.r_[self.U[N:], self.U[-N:]]
        U = np.clip(U, lo_v, hi_v)
        y_prev = U.copy()
        z = U.copy()
        dmax = cp.slew_per_step
        for _ in range(cp.fgm_iters):
            du = self.D @ z - b
            exc = np.sign(du) * np.maximum(np.abs(du) - dmax, 0)
            grad = 2 * (M @ z + q) + 2 * cp.slew_penalty * (self.D.T @ exc)
            y = np.clip(z - grad / L, lo_v, hi_v)
            z = y + beta * (y - y_prev)
            y_prev = y
        self.U = y_prev
        self.iters_used = cp.fgm_iters
        pred = c + G @ self.U
        return self.U[:N].copy(), pred.reshape(nh, N)
