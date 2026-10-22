"""B4 Unscented Kalman filter."""
from collections import deque

import numpy as np

from optics import lockin_ratios, drop_avg
from params import ControllerParams
from .common import N


class B4UKF:
    """Unscented Kalman filter over the four rings.

    State x = [delta (4), drift rate r (4), DC scale p (4)]
      delta  detuning from the ring's own laser (GHz)
      r      drift rate of the unmeasured heat, GHz/s. Random walk whose covariance is shaped as
             one common shift + two gradients (linear, quadratic across the row) + a small local term
      p      PD DC scale (laser power x coupling x PD gain), ADC full-scale units, slow random walk
    Process: delta(k+1) = delta + dh (reduced-order thermal model, known heater input) + Ts*r.
    Measurements, both through the exact data-averaged Lorentzian:
      zeta = y2/y1   curvature/slope ratio of the lock-in (proposal eq. 1): independent of optical
                     power and zero at the operating point whatever the dither gain, so it anchors the
                     absolute detuning without bias; slow (4 ms window, ~2 ms lag handled with the
                     estimate history) and noisy
      y0b            DC level of the latest 100 us block = p * L(delta + wobble): fast, needs p
    """

    def __init__(self, cp: ControllerParams, hwhm, mod, p_nominal):
        self.cp = cp
        self.hwhm = np.asarray(hwhm)
        self.mod = mod
        self.nx = 3 * N
        self.x = np.zeros(self.nx)
        self.x[2 * N:] = p_nominal
        self.P = np.diag(np.r_[np.full(N, 25.0), np.full(N, 1e4), (0.03 * np.asarray(p_nominal)) ** 2])
        e = np.ones(N) / 2
        g1 = np.linspace(-1, 1, N); g1 /= np.linalg.norm(g1)
        g2 = g1 ** 2; g2 -= g2.mean(); g2 /= np.linalg.norm(g2)
        self.Qr_base = ((cp.ukf_q_rate_common ** 2) * np.outer(e, e) * 4 / N
                        + (cp.ukf_q_rate_grad ** 2) * (np.outer(g1, g1) + np.outer(g2, g2))
                        + (cp.ukf_q_rate_local ** 2) * np.eye(N))
        self.qm = np.ones(N)            # per-ring adaptive multiplier of the drift-rate noise
        self.gated = np.zeros(N, int)   # consecutive DC samples rejected by the innovation gate
        self.ts = cp.ts
        n = self.nx
        alpha, beta, kappa = 0.5, 2.0, 0.0
        lam = alpha ** 2 * (n + kappa) - n
        self.c = n + lam
        self.wm = np.full(2 * n + 1, 1 / (2 * self.c))
        self.wc = self.wm.copy()
        self.wm[0] = lam / self.c
        self.wc[0] = lam / self.c + (1 - alpha ** 2 + beta)
        self.nis = np.zeros(N)
        self.nis_dc = np.zeros(N)
        self.hist = deque(maxlen=int(round(cp.lockin_window / cp.ts)))

    def init_ring(self, i, delta, var=9.0, p=None):
        for s in (i, N + i):
            self.P[s, :] = 0; self.P[:, s] = 0
        self.x[i] = delta
        self.x[N + i] = 0.0
        self.P[i, i] = var
        self.P[N + i, N + i] = 1e4
        if p is not None:
            s = 2 * N + i
            self.P[s, :] = 0; self.P[:, s] = 0
            self.x[s] = p
            self.P[s, s] = (0.03 * p) ** 2

    def predict(self, dh):
        ts = self.ts
        F = np.eye(self.nx)
        F[:N, N:2 * N] = ts * np.eye(N)
        self.x = F @ self.x
        self.x[:N] += dh
        Q = np.zeros((self.nx, self.nx))
        Q[:N, :N] = np.eye(N) * self.cp.ukf_q_delta ** 2
        dq = np.sqrt(self.qm)
        Q[N:2 * N, N:2 * N] = self.Qr_base * np.outer(dq, dq)
        Q[2 * N:, 2 * N:] = np.diag((self.cp.ukf_q_p * self.x[2 * N:]) ** 2)
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def _sigma(self):
        n = self.nx
        try:
            S = np.linalg.cholesky(self.c * self.P)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(self.P)
            self.P = (v * np.maximum(w, 1e-9)) @ v.T
            S = np.linalg.cholesky(self.c * self.P + np.eye(n) * 1e-9)
        return np.vstack([self.x, self.x + S.T, self.x - S.T])

    def update(self, m):
        """m: dict with zeta, rz, ok_lock (lock-in ratio), y0b, r0, ok_dc (block DC), a_nom
        (wobble GHz over the window), a_nom_dc (this block), cos_blk (n, 4) wobble phase samples
        of the last block, lag_lock / lag_dc (4,) estimated detuning change since the window
        centres, squelch (4,) bool."""
        il = np.where(m["ok_lock"])[0]
        idc = np.where(m["ok_dc"])[0]
        if len(il) == 0 and len(idc) == 0:
            return
        X = self._sigma()
        D, PS = X[:, :N], X[:, 2 * N:]
        mods = np.where(m["squelch"], 0.0, self.mod)
        cols = []
        if len(il):
            dl = D[:, il] - m["lag_lock"][None, il]
            zz = np.empty_like(dl)
            for mv in np.unique(mods[il]):
                c = np.where(mods[il] == mv)[0]
                q1, q2 = lockin_ratios(dl[:, c], m["a_nom"][il][c][None, :], self.hwhm[il][c][None, :], mv)
                zz[:, c] = q2 / q1
            cols.append(zz)
        if len(idc):
            dd = D[:, idc] - m["lag_dc"][None, idc]
            wob = m["wob_blk"][None, :, idc]
            lv = drop_avg(dd[:, None, :] + wob, self.hwhm[idc][None, None, :], mods[idc][None, None, :])
            cols.append(PS[:, idc] * lv.mean(1))
        Z = np.hstack(cols)
        z = np.r_[m["zeta"][il], m["y0b"][idc]]
        rr = np.r_[m["rz"][il], m["r0"][idc]]
        zm = self.wm @ Z
        dZ = Z - zm
        var = (self.wc[:, None] * dZ * dZ).sum(0) + rr
        nis_all = (z - zm) ** 2 / var
        # robust (Huber-style) weighting: an innovation beyond the gate gets its noise inflated in
        # proportion, so squelch transients, glitches or a ring that has left its slope cannot drag
        # the shared state, while a genuine fast disturbance still gets through (and raises qm)
        nl = len(il)
        big = nis_all > self.cp.ukf_gate
        gd = np.zeros(N, bool)
        gd[idc[big[nl:]]] = True
        self.gated = np.where(gd, self.gated + 1, np.where(m["ok_dc"], 0, self.gated))
        nisd = np.zeros(N)
        nisd[idc] = nis_all[nl:]
        self.nis_dc = np.where(m["ok_dc"], 0.7 * self.nis_dc + 0.3 * np.minimum(nisd, 4 * self.cp.ukf_gate), self.nis_dc)
        nis = np.zeros(N)
        nis[il] = nis_all[:nl]
        self.nis = np.where(m["ok_lock"], 0.9 * self.nis + 0.1 * np.minimum(nis, 50), self.nis)
        # adaptive drift-rate noise per ring: persistent large DC innovations -> faster disturbance
        target = np.clip(self.nis_dc / 3.0, 1.0, self.cp.ukf_q_adapt)
        self.qm = np.where(target > self.qm, target, 0.98 * self.qm + 0.02 * target)
        rr = rr * np.maximum(1.0, nis_all / self.cp.ukf_gate)
        dX = X - self.x
        Pzz = (self.wc[:, None] * dZ).T @ dZ + np.diag(rr)
        Pxz = (self.wc[:, None] * dX).T @ dZ
        K = np.linalg.solve(Pzz, Pxz.T).T
        # an outlying ring may only correct its own states, never its neighbours'
        rows_ring = np.r_[il, idc]
        state_ring = np.tile(np.arange(N), 3)
        for j in np.where(big)[0]:
            K[state_ring != rows_ring[j], j] = 0.0
        self.x = self.x + K @ (z - zm)
        # Joseph-free covariance update is only valid for the optimal gain; recompute generally
        self.P = self.P - K @ Pxz.T - Pxz @ K.T + K @ Pzz @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    @property
    def q_mult(self):
        return float(self.qm.max())

    def remember(self):
        self.hist.append(self.x[:N].copy())

    @property
    def delta(self):
        return self.x[:N]

    @property
    def rate(self):
        return self.x[N:2 * N]

    @property
    def pscale(self):
        return self.x[2 * N:]

    @property
    def sigma(self):
        return np.sqrt(np.maximum(np.diag(self.P)[:N], 0))
