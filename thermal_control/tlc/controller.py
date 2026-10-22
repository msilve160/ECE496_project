"""The TLC top level: wires blocks B1-B10 and runs one control step; plus the baseline PI."""
from collections import deque

import numpy as np

from optics import lockin_ratios
from params import ControllerParams, PlantParams
from .b1_dither import B1Dither
from .b2_conditioner import B2Conditioner
from .b3_lockin import B3LockIn
from .b4_thermal_model import ThermalModel
from .b4_ukf import B4UKF
from .b5_supervisor import SupervisorMixin
from .b6_setpoints import B6Setpoints
from .b7_mpc import B7MPC
from .b7_pi import PIController
from .b8_output import B8Output
from .b9_diagnostics import B9Diagnostics
from .b10_calibration import B10Calibration
from .common import Mode, N


class TLC(SupervisorMixin):
    """Thermal locking controller: call step() once per 100 us block of PD samples."""

    def __init__(self, pp: PlantParams, cp: ControllerParams, use_mpc=True, learn=True):
        self.pp, self.cp = pp, cp
        self.cal = B10Calibration(pp, cp)
        self.sp = B6Setpoints(self.cal, cp)
        self.dither = B1Dither(cp)
        self.cond = B2Conditioner(cp, pp.pd_rate)
        self.out = B8Output(cp)
        self.out_delay = 0.5 / cp.out_rate            # ZOH of the dither samples
        self.lockin = B3LockIn(cp, self.out_delay, pp.pd_rate / cp.decim)
        self.ukf = B4UKF(cp, self.cal.hwhm, self.cal.mod,
                         self.cal.p_laser * self.cal.depth / pp.pd_full_scale_mw)
        self.theta_obs = np.ones(N)
        self.model = ThermalModel(cp.ts, pp.tau_hs, pp.tau_x)
        self.mpc = B7MPC(cp)
        self.pi = PIController(cp, self.cal.dc_gain_open())
        self.diag = B9Diagnostics()
        self.use_mpc = use_mpc
        self.learn = learn
        self.modes = np.full(N, Mode.INIT)
        self.fault = False
        self.k = 0
        self.t = 0.0
        self.h_hist = deque(maxlen=int(round(cp.lockin_window / cp.ts)))
        self.a_hist = deque(maxlen=int(round(cp.lockin_window / cp.ts)))
        self.u_cmd = np.zeros(N)
        self.lol_count = np.zeros(N)
        self.lim_count = np.zeros(N)
        self.limit_flag = np.zeros(N, bool)
        self.invalid_count = np.zeros(N)
        self.valid_count = np.zeros(N)
        self.squelch = np.zeros(N, bool)
        self.sq_count = np.zeros(N)
        self.acq = [dict() for _ in range(N)]
        self.hold_from = np.full(N, Mode.TRACK)
        self._refresh_model()
        # acquisition lookup: z1 / max(z1) on the approach -> detuning on the long side
        self._build_acq_tables()
        self.fallback_steps = []
        self.iq_avg = np.zeros(N, complex)
        self.theta_obs = np.ones(N)
        self._d_blk = np.zeros(N)
        self.log = None

    # ------------------------------------------------------------------------------
    def _refresh_model(self):
        gf, ghs, gx = self.cal.gains(self.sp.targets)
        self.model.set_gains(gf, ghs, gx)
        self.mpc.build(self.model)
        H = self.cal.dither_response(self.sp.targets, self.cp.tone_freqs, self.out_delay)
        if hasattr(self, "H") and np.any(np.abs(getattr(self, "iq_avg", np.zeros(N))) > 0.5):
            own = np.arange(N)
            H[own, own] = np.abs(H[own, own]) * np.exp(1j * np.angle(self.H[own, own]))
        self.H = H
        self.h_own_abs = np.abs(np.diag(self.H))
        self.pi.g = self.cal.dc_gain_open() * np.real(self.cal.self_heating_factor(self.sp.targets))
        # slope of z1 at target per GHz (for the PI error signal)
        a = self.cp.dither_nominal
        d = 0.05
        z_hi, _ = lockin_ratios(self.sp.targets + d, a, self.cal.hwhm, self.cal.mod)
        z_lo, _ = lockin_ratios(self.sp.targets - d, a, self.cal.hwhm, self.cal.mod)
        self.z1_slope = (z_hi - z_lo) / (2 * d) / a
        self.z1_target, _ = lockin_ratios(self.sp.targets, a, self.cal.hwhm, self.cal.mod)
        self.z1_target = self.z1_target / a

    def _build_acq_tables(self):
        self.acq_tab = []
        for i in range(N):
            dgrid = np.linspace(-150, 60, 4201)
            z1, _ = lockin_ratios(dgrid, self.cp.dither_max, self.cal.hwhm[i], self.cal.mod)
            zmax = z1.max()
            self.acq_tab.append((dgrid, z1 / zmax))
            self.acq_peak = getattr(self, "acq_peak", []) + [zmax / self.cp.dither_max]
        # lock threshold: ratio at the target (negative) times 0.85
        self.acq_ratio = np.array([np.interp(self.sp.targets[i], *self.acq_tab[i]) for i in range(N)])
        self.acq_peak = np.array(self.acq_peak)

    def _guard_hi(self):
        """B8 channel guard (R23): max command keeping the predicted resonance within
        +channel_guard GHz of the ring's own laser target."""
        g_self = self.model.gf + self.model.ghs
        hi = np.ones(N)
        tr = self.modes == Mode.TRACK
        room = self.sp.targets + self.cp.channel_guard - self.ukf.delta
        hi_tr = self.u_cmd + room / g_self
        hi = np.where(tr, np.clip(hi_tr, 0, 1), hi)
        # acquisition: bound from calibration (unheated position at reference temperature)
        acq = (self.modes == Mode.ACQUIRE) | (self.modes == Mode.REACQUIRE)
        g_open = self.cal.dc_gain_open()
        hi_acq = (self.sp.targets + self.cp.channel_guard - self.cal.unheated) / g_open
        hi = np.where(acq, np.clip(hi_acq, 0, 1), hi)
        return hi

    # ------------------------------------------------------------------------------
    def warm_start(self, u0, delta_true):
        """Begin already locked (used for disturbance tests): thermal model and output at u0."""
        self.u_cmd = u0.copy()
        self.out.u_prev = u0.copy()
        self.out.last = u0.copy()
        self.model.settle(u0)
        self.modes[:] = Mode.TRACK
        for i in range(N):
            self.ukf.init_ring(i, delta_true[i], var=1.0)
        self.ukf.P[N:2 * N, N:2 * N] = np.eye(N) * 100.0
        self.ukf.hist.clear()
        self.pi.reset(u0)
        self.h_hist.clear()
        for _ in range(self.h_hist.maxlen):
            self.h_hist.append(self.model.h(u0))
        self.dither.schedule(self.ukf.sigma, self.modes)
        self.dither.set_command_amplitude(self.h_own_abs, np.ones(N))
        self._d_blk = self.dither.cmd_amp.copy()

    def cold_start(self, rings=range(N)):
        for i in rings:
            self.modes[i] = Mode.ACQUIRE
            self.acq[i] = dict(zmax=0.0, zf=0.0, n=0)
        self.dither.schedule(self.ukf.sigma, self.modes)
        self.dither.set_command_amplitude(self.h_own_abs, np.ones(N))
        self._d_blk = self.dither.cmd_amp.copy()

    # ------------------------------------------------------------------------------
    def step(self, pd, t_block_start):
        """Consume one block of PD samples, return the heater command sequence for the next block."""
        cp = self.cp
        ts = cp.ts
        self.t = t_block_start + ts
        # ---- B2 ------------------------------------------------------------------
        y, invalid = self.cond.process(pd)
        n_dec = y.shape[0]
        t_dec = t_block_start + (np.arange(n_dec) * cp.decim) / self.pp.pd_rate - self.cond.delay
        # ---- B3 ------------------------------------------------------------------
        self.lockin.push(y, t_dec, invalid)
        y0, y1, y2, y1_all, valid = self.lockin.demod(self.H)
        # B10 online phase calibration of each own tone (slope sign known on the locked side)
        trk = (self.modes == Mode.TRACK) & valid
        if trk.any():
            c = self.lockin.own_iq * np.sign(self.z1_target)
            self.iq_avg = np.where(trk, 0.98 * self.iq_avg + 0.02 * c / np.maximum(np.abs(c), 1e-9), self.iq_avg)
            ph = np.angle(self.iq_avg)
            use = np.abs(self.iq_avg) > 0.5
            if use.any():
                d = np.where(use, ph - np.angle(np.diag(self.H)), 0.0)
                self.H = self.H.copy()
                self.H[np.arange(N), np.arange(N)] *= np.exp(1j * d)
        z1 = y1 / np.maximum(y0, 1e-6)
        z2 = y2 / np.maximum(y0, 1e-6)
        y0b = y.mean(0)                                   # fast DC level of this block
        # known detuning wobble in this block from every heater's tone (own + crosstalk), GHz
        ph_t = self.lockin.w[None, :] * (t_dec[:, None] - self.out_delay)          # (n, tones)
        wob_blk = np.real(np.exp(1j * ph_t) @ (self.H * self._d_blk[None, :]).T)   # (n, rings)
        a_win = np.mean(np.array(self.a_hist), axis=0) if len(self.a_hist) else self.dither.amp_ghz
        # measurement noise from B2's noise estimate (quality score)
        sig_f = self.cond.noise * self.cond.noise_gain
        r_lock = (sig_f * np.sqrt(2.0 / self.lockin.nwin) * 1.5 / np.maximum(y0, 1e-3)) ** 2
        z1s = np.where(np.abs(z1) > 1e-6, z1, 1e-6)
        zeta = z2 / z1s
        infl = 1.0 + 4.0 * (self.ukf.qm - 1.0)           # lagged channel is less reliable in transients
        # successive windows overlap by (window - Ts)/window, so each update carries only ~Ts/window
        # of new information: inflate R accordingly
        overlap = self.lockin.nwin / n_dec
        rz = (r_lock / z1s ** 2 + (zeta ** 2) * r_lock / z1s ** 2 + (0.05 * zeta) ** 2 + (3e-4) ** 2) * infl * overlap
        # quality: the slope must be clearly present for the ratio to be meaningful
        z1_exp = np.abs(self.z1_target) * a_win
        lock_q = np.abs(z1) > 0.4 * z1_exp
        r0 = (sig_f / np.sqrt(n_dec)) ** 2 * 2.0 + (1.5e-3 * y0b) ** 2

        # squelch recognition from the PD alone: data-pattern noise disappears
        self._update_squelch()

        # ---- B4 predict with the thermal model (commands applied during this block) --
        h_prev = self.h_hist[-1] if len(self.h_hist) else self.model.h(self.out.u_prev)
        u_mean = self._u_mean_last if hasattr(self, "_u_mean_last") else self.u_cmd
        h_now = self.model.step(u_mean, self.out.u_prev)
        dh = h_now - h_prev
        self.h_hist.append(h_now)
        tracking = (self.modes == Mode.TRACK)
        self.ukf.predict(np.where(tracking | (self.modes == Mode.HOLD), dh, 0.0))
        prior = self.ukf.delta.copy()
        hist = np.array(self.ukf.hist) if len(self.ukf.hist) else prior[None, :]
        lag_lock = prior - hist.mean(0)
        lag_dc = 0.5 * (prior - hist[-1])
        meas = dict(zeta=zeta, rz=rz, ok_lock=valid & tracking & lock_q,
                    y0b=y0b, r0=r0, ok_dc=~invalid & tracking, a_nom=a_win,
                    wob_blk=wob_blk, lag_lock=lag_lock, lag_dc=lag_dc, squelch=self.squelch)
        p_before = self.ukf.x[2 * N:].copy()
        self.ukf.update(meas)
        # squelch changes the lineshape, not the optical power or PD gain: keep the DC scale fixed so
        # the model mismatch of the bare Lorentzian cannot be absorbed into p as a detuning bias
        sq = self.squelch & tracking
        self.ukf.x[2 * N:] = np.where(sq, p_before, self.ukf.x[2 * N:])
        # a held ring has no measurement: propagate it on the model alone, with the common drift
        # rate seen by the tracking rings
        held = ~tracking
        # physical bound on the drift rate: D2's peak rate is 2300 K/s = 34.5 GHz/ms
        self.ukf.x[N:2 * N] = np.clip(self.ukf.x[N:2 * N], -self.cp.rate_max, self.cp.rate_max)
        if held.any():
            common = self.ukf.rate[tracking].mean() if tracking.any() else 0.0
            for i in np.where(held)[0]:
                self.ukf.x[i] = prior[i]
                self.ukf.x[N + i] = common
        self.ukf.remember()
        upd = valid & tracking

        # ---- B10 learning ---------------------------------------------------------
        # heater efficiency at the tone: measured slope vs. the model's slope at the estimate
        zm1, _ = lockin_ratios(self.ukf.delta - lag_lock, a_win, self.cal.hwhm,
                               np.where(self.squelch, 0.0, self.cal.mod))
        th_obs = np.where(np.abs(zm1) > 1e-6, z1 / np.where(np.abs(zm1) > 1e-6, zm1, 1.0), 1.0)
        ok_th = upd & lock_q & ~self.squelch & (self.ukf.sigma < 1.0) & (self.ukf.q_mult < 2.0)
        self.theta_obs = np.where(ok_th, 0.995 * self.theta_obs + 0.005 * th_obs, self.theta_obs)
        if self.learn:
            self.cal.rls_efficiency(np.clip(th_obs, 0.5, 1.6), ok_th)
            # coupling from cross tones: y1_all[i, j] for neighbours j
            if np.all(upd) and not self.squelch.any() and self.ukf.q_mult < 2.0:
                yy, pp_ = [], []
                for i in range(N):
                    for j in (i - 1, i + 1):
                        if 0 <= j < N:
                            # expected cross response relative to own: |H_ij| d_j / (|H_ii| d_i)
                            ratio = (np.abs(self.H[i, j]) * self.dither.cmd_amp[j]) / \
                                    (np.abs(self.H[i, i]) * self.dither.cmd_amp[i])
                            yy.append(y1_all[i, j])
                            pp_.append(y1[i] * ratio / max(self.theta_obs[i], 0.5) * self.theta_obs[j])
                self.cal.rls_coupling(np.array(yy), np.array(pp_), True)
            if self.k % 500 == 499 and self.cal.commit():
                ghs0, gx0 = self.model.ghs.copy(), self.model.gx.copy()
                self._refresh_model()
                # keep the model's heater contribution continuous across the gain change
                self.model.z_hs *= self.model.ghs / ghs0
                xs_old = gx0 @ self.out.u_prev
                xs_new = self.model.gx @ self.out.u_prev
                self.model.z_x *= np.where(np.abs(xs_old) > 1e-9, xs_new / np.where(np.abs(xs_old) > 1e-9, xs_old, 1), 1)
                self.h_hist[-1] = self.model.h(self.out.u_prev)
                h_now = self.h_hist[-1]      # the MPC below must see the same model as its matrices

        # ---- B5 supervisor ---------------------------------------------------------
        self._supervise(z1, y0b, valid, invalid)

        # ---- B6 targets, B7 control -------------------------------------------------
        target = self.sp.targets
        guard_hi = self._guard_hi()
        lo = np.zeros(N)
        hi = guard_hi.copy()
        planned = self.u_cmd.copy()
        qmask = (self.modes == Mode.TRACK).astype(float)
        for i in range(N):
            if self.modes[i] in (Mode.ACQUIRE, Mode.REACQUIRE, Mode.HOLD, Mode.INIT):
                planned[i] = self._planned_command(i)
                lo[i] = hi[i] = planned[i]
        if self.fault or not self.use_mpc:
            err = (z1 / np.maximum(self.dither.amp_ghz, 1e-3) - self.z1_target) / self.z1_slope
            u_pi = self.pi.step(err, valid & tracking, guard_hi)
            u_new = np.where(tracking, u_pi, planned)
            self.pi.u = u_new.copy(); self.pi.integ = np.where(tracking, self.pi.integ, u_new)
            pred = None
        else:
            u_new, pred = self.mpc.solve(self.ukf.delta, self.ukf.rate, h_now, self.u_cmd,
                                         target, qmask, lo, np.maximum(hi, lo))
        self._limit_flags(u_new, guard_hi)

        # ---- B1 dither schedule, B8 output ----------------------------------------------
        self.dither.schedule(self.ukf.sigma, self.modes)
        self.dither.set_command_amplitude(self.h_own_abs, np.ones(N))
        self._d_blk = self.dither.cmd_amp.copy()
        self.a_hist.append(self.dither.cmd_amp * self.h_own_abs)
        self._d_blk = self.dither.cmd_amp.copy()
        seq, u_c, u_mean = self.out.produce(u_new, t_block_start + ts, self.dither, guard_hi)
        self._u_mean_last = u_mean
        self.u_cmd = u_c
        self.k += 1
        if self.k % 100 == 0:
            self.diag.status(self.t, self.modes, self.ukf.delta - target, self.u_cmd)
        self.last = dict(z1=z1, z2=z2, y0=y0, valid=valid, delta_hat=self.ukf.delta.copy(),
                         rate_hat=self.ukf.rate.copy(), theta=self.theta_obs.copy(),
                         sigma=self.ukf.sigma.copy(), modes=self.modes.copy(), u=u_c.copy(),
                         amp=self.dither.amp_ghz.copy(), squelch=self.squelch.copy(),
                         fault=self.fault, limit=self.limit_flag.copy(), eff=self.cal.eff.copy(),
                         cpl=self.cal.cpl, qmult=self.ukf.q_mult)
        return seq


class BaselinePI(TLC):
    """Baseline (App. C.3): independent per-ring dither lock-in + PI, fixed offline target map,
    no coupling model, no disturbance estimate, no learning. Shares B1/B2/B3/B8 with the TLC."""

    def __init__(self, pp, cp):
        super().__init__(pp, cp, use_mpc=False, learn=False)

    def step(self, pd, t_block_start):
        cp = self.cp
        ts = cp.ts
        self.t = t_block_start + ts
        y, invalid = self.cond.process(pd)
        t_dec = t_block_start + (np.arange(y.shape[0]) * cp.decim) / self.pp.pd_rate - self.cond.delay
        self.lockin.push(y, t_dec, invalid)
        y0, y1, y2, y1_all, valid = self.lockin.demod(self.H)
        z1 = y1 / np.maximum(y0, 1e-6)
        err = (z1 / cp.dither_nominal - self.z1_target) / self.z1_slope
        hi = np.ones(N)
        u_new = self.pi.step(err, valid, hi)
        self.dither.schedule(None, None, fixed=cp.dither_nominal)
        self.dither.set_command_amplitude(self.h_own_abs, np.ones(N))
        seq, u_c, _ = self.out.produce(u_new, t_block_start + ts, self.dither, hi)
        self.u_cmd = u_c
        self.k += 1
        self.last = dict(z1=z1, y0=y0, valid=valid, delta_hat=self.sp.targets + np.where(valid, err, 0),
                         u=u_c.copy(), modes=self.modes.copy(), amp=self.dither.amp_ghz.copy(),
                         fault=False, squelch=np.zeros(N, bool), limit=np.zeros(N, bool))
        return seq
