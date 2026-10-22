"""B10 Calibration store + recursive least squares refinement."""
import numpy as np

from optics import drop_avg, inflection
from params import ControllerParams, PlantParams, coupling_matrix
from .common import N


class B10Calibration:
    """Stores the controller's model of the device and refines it during normal operation.

    The stored values are the nominal design values (what an offline characterisation would
    give). The plant may differ (heater_eff_scale, coupling_scale); B10 learns:
      * heater efficiency per ring, from the UKF's dither-gain state (scalar RLS, forgetting)
      * one thermal-coupling scale, from the lock-in response of each ring to its neighbours'
        dither tones (the dither is an independent excitation, so this is not biased by the loop)
    """

    def __init__(self, pp: PlantParams, cp: ControllerParams):
        self.pp, self.cp = pp, cp
        self.kappa = pp.kappa
        self.hwhm = np.array(pp.hwhm)
        self.depth = np.array(pp.depth)
        self.p_laser = np.array(pp.laser_power)
        self.mod = pp.mod_shift
        self.unheated = np.array(pp.unheated_offset, float)
        self.cnom = coupling_matrix(pp.coupling)
        self.eff = np.ones(N)          # learned heater efficiency scale
        self.cpl = 1.0                 # learned coupling scale
        # RLS states
        self._eff_P = np.full(N, 0.01)
        self._eff_x = np.ones(N)
        self._c_num = 0.0
        self._c_den = 0.0
        self.version = 0
        self.targets_raw = np.array([inflection(h, self.mod, +1) for h in self.hwhm])

    # ---- thermal model gains (GHz per unit command) -----------------------------------
    def self_heating_factor(self, delta, omega=0.0):
        """Complex closed-loop factor of optical self heating at detuning `delta`.

        d(delta) = d(delta_lin) / (1 - K L'(delta) * F(omega)), with K = kappa * R_sh * abs * P.
        On the long-wavelength side (delta > 0) L' < 0, so self heating is negative feedback.
        """
        pp = self.pp
        eps = 1e-3
        dl = (drop_avg(delta + eps, self.hwhm, self.mod) - drop_avg(delta - eps, self.hwhm, self.mod)) / (2 * eps)
        k = self.kappa * pp.abs_frac * self.p_laser * 1e-3 * (pp.r_shf + pp.r_shs)
        wf = pp.r_shf / (pp.r_shf + pp.r_shs)
        frq = wf / (1 + 1j * omega * pp.tau_shf) + (1 - wf) / (1 + 1j * omega * pp.tau_shs)
        return 1.0 / (1.0 - k * dl * frq)

    def gains(self, targets):
        """Fast / slow self gains (4,), crosstalk gain matrix (4,4), all in GHz per unit u."""
        pp = self.pp
        sh = np.real(self.self_heating_factor(targets))
        base = self.kappa * pp.p_heater_max * self.eff
        gf = base * pp.r_hf * sh
        ghs = base * pp.r_hs * sh
        gx = (self.kappa * pp.p_heater_max * (pp.r_hf + pp.r_hs)) * self.cpl * self.cnom \
            * self.eff[None, :] * sh[:, None]
        return gf, ghs, gx

    def dc_gain_open(self):
        """Self DC gain far from resonance (no self heating), GHz per unit u (used by Acquire)."""
        pp = self.pp
        return self.kappa * pp.p_heater_max * (pp.r_hf + pp.r_hs) * self.eff

    def dither_response(self, targets, tones, delay):
        """Complex response H[i, j] of ring i's detuning to heater j's tone j (GHz per unit u)."""
        pp = self.pp
        h = np.zeros((N, N), complex)
        for j, f in enumerate(tones):
            w = 2 * np.pi * f
            sh = self.self_heating_factor(targets, w)
            dly = np.exp(-1j * w * delay)
            own = self.kappa * pp.p_heater_max * self.eff[j] * (pp.r_hf / (1 + 1j * w * pp.tau_hf)
                                                               + pp.r_hs / (1 + 1j * w * pp.tau_hs))
            h[j, j] = own * sh[j] * dly
            for i in range(N):
                if i != j:
                    h[i, j] = (self.kappa * pp.p_heater_max * (pp.r_hf + pp.r_hs) * self.cpl * self.cnom[i, j]
                               * self.eff[j] / (1 + 1j * w * pp.tau_x)) * sh[i] * dly
        return h

    # ---- RLS refinement ----------------------------------------------------------------
    def rls_efficiency(self, theta_obs, ok):
        """Scalar RLS (random-constant model with forgetting) on the UKF's dither-gain estimate."""
        lam = self.cp.rls_lambda
        for i in range(N):
            if not ok[i]:
                continue
            target = self.eff[i] * theta_obs[i]        # theta is relative to the current model
            p = self._eff_P[i] / lam
            k = p / (p + 1.0)
            self._eff_x[i] += k * (target - self._eff_x[i])
            self._eff_P[i] = (1 - k) * p

    def rls_coupling(self, y, phi, ok):
        """Exponentially weighted least squares for the coupling scale: y ~ c * phi."""
        lam = self.cp.rls_lambda
        if not ok:
            return
        self._c_num = lam * self._c_num + float(np.sum(phi * y))
        self._c_den = lam * self._c_den + float(np.sum(phi * phi))

    def commit(self):
        """Apply learned values (bounded) to the model. Returns True if anything changed."""
        lo, hi = self.cp.eff_bounds
        new_eff = np.clip(self._eff_x, lo, hi)
        changed = np.max(np.abs(new_eff - self.eff)) > 0.005
        self.eff = new_eff
        if self._c_den > 0:
            c_hat = self.cpl * self._c_num / self._c_den
            lo, hi = self.cp.coupling_bounds
            c_new = float(np.clip(c_hat, lo, hi))
            if abs(c_new - self.cpl) > 0.01:
                changed = True
                # restart the accumulators relative to the new scale
                self._c_num *= self.cpl / c_new
            self.cpl = c_new
        if changed:
            self.version += 1
        return changed
