"""B5 Supervisor: state machine, squelch detection, loss of lock, fallback, limit flags."""
import numpy as np

from optics import drop_avg
from .common import Mode, N


class SupervisorMixin:
    """B5 supervisor: per-ring mode machine (INIT/ACQUIRE/TRACK/HOLD/REACQUIRE/FAULT), squelch
    recognition, loss-of-lock detection, fallback entry and heater-limit flags (R20).
    Mixed into `TLC`, whose state it reads and updates."""

    def start_acquire(self, i):
        self.modes[i] = Mode.ACQUIRE
        self.acq[i] = dict(zmax=0.0, zf=0.0, n=0)

    def force_fault(self):
        self._enter_fault()

    def _enter_fault(self):
        if not self.fault:
            self.fault = True
            self.pi.reset(self.u_cmd)        # bumpless transfer (R22)
            self.diag.alarm(self.t, -1, "fallback")

    def _update_squelch(self):
        # expected data-pattern noise at lock is well above the TIA noise floor. Hysteresis: on after
        # 2 quiet blocks below 1.25x the floor, off only after 2 blocks above 2x the floor
        floor = self.pp.pd_noise_fs
        quiet = self.cond.noise_inst < 1.25 * floor
        loud = self.cond.noise_inst > 2.0 * floor
        self.sq_count = np.where(quiet, self.sq_count + 1, 0)
        self.unsq_count = np.where(loud, getattr(self, "unsq_count", np.zeros(N)) + 1, 0)
        tracking = self.modes == Mode.TRACK
        on = np.where(self.squelch, self.unsq_count < 2, self.sq_count >= 2)
        self.squelch = np.where(tracking, on, False)

    def _planned_command(self, i):
        cp = self.cp
        st = self.acq[i]
        if self.modes[i] == Mode.ACQUIRE:
            step = cp.acquire_rate * 1e3 * cp.ts / self.cal.dc_gain_open()[i]
            return min(self.u_cmd[i] + step, self._guard_hi()[i])
        if self.modes[i] == Mode.REACQUIRE:
            if st.get("phase", "back") == "back":
                return max(self.u_cmd[i] - 0.9 * self.cp.slew_per_step, st["u_floor"])
            step = cp.reacquire_rate * 1e3 * cp.ts / self.cal.dc_gain_open()[i]
            return min(self.u_cmd[i] + step, self._guard_hi()[i])
        return self.u_cmd[i]     # HOLD / INIT: keep last safe command

    def _supervise(self, z1, y0, valid, invalid):
        cp = self.cp
        for i in range(N):
            m = self.modes[i]
            # sensor validity -> HOLD (R21)
            if m in (Mode.TRACK, Mode.ACQUIRE, Mode.REACQUIRE) and not valid[i] and invalid[i]:
                self.hold_from[i] = m
                self.modes[i] = Mode.HOLD
                # decouple the held ring so the other rings' updates cannot drag its estimate
                self.ukf.init_ring(i, self.ukf.x[i], var=4.0, p=self.ukf.x[2 * N + i])
                self.valid_count[i] = 0
                self.diag.alarm(self.t, i, "pd_invalid")
                continue
            if m == Mode.HOLD:
                self.valid_count[i] = self.valid_count[i] + 1 if valid[i] else 0
                if self.valid_count[i] >= 3:
                    if self.hold_from[i] == Mode.TRACK:
                        self.modes[i] = Mode.TRACK
                        self.ukf.init_ring(i, self.ukf.x[i], var=9.0, p=self.ukf.x[2 * N + i])
                        self.ukf.hist.clear()
                    else:
                        self.start_acquire(i)
                continue
            if m in (Mode.ACQUIRE, Mode.REACQUIRE):
                st = self.acq[i]
                if m == Mode.REACQUIRE and st.get("phase") == "back":
                    if self.u_cmd[i] <= st["u_floor"] + 1e-6:
                        st.update(phase="ramp", zmax=0.0, zf=0.0, n=0)
                    continue
                if not valid[i]:
                    continue
                zn = z1[i] / self.dither.amp_ghz[i]
                st["zf"] = 0.6 * st.get("zf", 0.0) + 0.4 * zn
                st["zmax"] = max(st.get("zmax", 0.0), st["zf"])
                st["n"] = st.get("n", 0) + 1
                # the long-wavelength flank: normalised slope (z1 / wobble) has the opposite sign to
                # the approach and reaches ratio * peak at the operating point. Reference the peak to the
                # larger of the observed and the modelled one so a ring that starts inside the dip works.
                dg, ratio = self.acq_tab[i]
                peak = self.acq_peak[i]
                if st["zf"] <= 0.85 * self.acq_ratio[i] * peak:
                    sel = (dg > 0) & (dg < 2 * self.cal.hwhm[i])
                    est = float(np.interp(-st["zf"] / peak, -ratio[sel], dg[sel]))
                    est += cp.acquire_rate * 1e3 * 0.5 * cp.lockin_window      # lock-in lag
                    lv = drop_avg(est, self.cal.hwhm[i], self.cal.mod)
                    self.ukf.init_ring(i, est, var=9.0, p=y0[i] / max(lv, 0.05))
                    self.ukf.hist.clear()
                    self.modes[i] = Mode.TRACK
                    self.lol_count[i] = 0
                    self.diag.alarm(self.t, i, "locked")
                elif self.u_cmd[i] >= self._guard_hi()[i] - 1e-6 and st["n"] > 20:
                    self.diag.alarm(self.t, i, "acquire_failed")
                    self.modes[i] = Mode.FAULT
                continue
            if m == Mode.TRACK:
                err = self.ukf.delta[i] - self.sp.targets[i]
                wrong_side = valid[i] and z1[i] > 0.25 * self.dither.amp_ghz[i] / self.cal.hwhm[i]
                bad = abs(err) > cp.lol_err or wrong_side or self.ukf.gated[i] > 20
                self.lol_count[i] = self.lol_count[i] + 1 if bad else 0
                if self.lol_count[i] * cp.ts >= cp.lol_time:
                    self.diag.alarm(self.t, i, "loss_of_lock")
                    # decouple the lost ring so its runaway estimate cannot leak into its neighbours
                    self.ukf.init_ring(i, self.sp.targets[i], var=100.0, p=self.ukf.x[2 * N + i])
                    back = cp.reacquire_backoff / self.cal.dc_gain_open()[i]
                    self.acq[i] = dict(phase="back", u_floor=max(self.u_cmd[i] - back, 0.0))
                    self.modes[i] = Mode.REACQUIRE
                    self.lol_count[i] = 0

    def _limit_flags(self, u_new, guard_hi):
        """R20: flag when a heater limit blocks the correction."""
        err = self.ukf.delta - self.sp.targets
        at_lo = (u_new <= 1e-4) & (err > 2.0)
        at_hi = (u_new >= guard_hi - 1e-4) & (err < -2.0)
        blocked = (at_lo | at_hi) & (self.modes == Mode.TRACK)
        self.lim_count = np.where(blocked, self.lim_count + 1, 0)
        newflag = (self.lim_count * self.cp.ts >= self.cp.limit_flag_time)
        for i in np.where(newflag & ~self.limit_flag)[0]:
            self.diag.alarm(self.t, i, "heater_limit")
        self.limit_flag = newflag
