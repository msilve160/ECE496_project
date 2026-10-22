"""Coupled thermal-optical plant: four cascaded microring modulators on one bus.

Per ring i the ring temperature rise is the sum of six first-order contributions
(structure and constants from ref [4], extended to four coupled rings):
  HF, HS  fast / slow heater paths (own heater)
  X       thermal crosstalk from the other three heaters through the substrate
  SF, SS  fast / slow optical self heating from absorbed laser power (nonlinear in detuning)
  E       modulator electrical self heating (removed during squelch)
plus the external disturbance temperature dT (D1-D3). The resonance detuning from the ring's own
laser is
  delta_i = unheated_i + kappa * (HF+HS+X+SF+SS+E+dT)_i - laser_shift_i            [GHz]
The monitor PD is a drop-port (in-ring) tap whose power follows the data-averaged Lorentzian of
every laser on the bus (own channel plus tails of the neighbours, FSR-periodic), with NRZ
pattern noise, TIA noise and a 12-bit ADC.

The plant integrates at 1 us (the PD rate) with heater commands held for 10 us (S2 at 100 kS/s).
"""
import numpy as np

from optics import drop_avg
from params import PlantParams, coupling_matrix, N_RINGS


class Plant:
    def __init__(self, p: PlantParams, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.dt = 1.0 / p.pd_rate
        self.hwhm = np.array(p.hwhm)
        self.depth = np.array(p.depth)
        self.p_laser = np.array(p.laser_power)
        self.unheated = np.array(p.unheated_offset, float)
        self.eff = np.array(p.heater_eff_scale)
        self.cmat = coupling_matrix(p.coupling) * p.coupling_scale
        r_self = p.r_hf + p.r_hs
        # first-order paths: (time constant, state index)
        self.taus = np.array([p.tau_hf, p.tau_hs, p.tau_x, p.tau_shf, p.tau_shs, p.tau_e])
        self.r_self = r_self
        self.state = np.zeros((6, N_RINGS))   # K
        self.t = 0.0
        self.sigma_ones = 0.5 / np.sqrt(p.sym_rate * self.dt)
        # channel offsets of every laser relative to ring i's own laser (GHz, wavelength direction)
        idx = np.arange(N_RINGS)
        self.ch_off = (idx[None, :] - idx[:, None]) * p.channel_spacing   # [i, j]
        self.last_delta = self.unheated.copy()

    # ------------------------------------------------------------------------------------
    def settle(self, u, dist_sample, n_iter=200):
        """Put the thermal states at steady state for constant command u (used for warm starts)."""
        p = self.p
        for _ in range(n_iter):
            ph = u * p.p_heater_max * self.eff
            delta = self._delta(self.state.sum(0), dist_sample["dT"][0], dist_sample["laser"][0])
            pabs = self._absorbed(delta, dist_sample["pscale"][0], dist_sample["squelch"][0])
            tgt = self._targets(ph, pabs, dist_sample["squelch"][0])
            self.state = tgt
        self.last_delta = self._delta(self.state.sum(0), dist_sample["dT"][0], dist_sample["laser"][0])

    def _delta(self, temp, dT, laser):
        return self.unheated + self.p.kappa * (temp + dT) - laser

    def _lineshape_sum(self, delta, frac_ones, squelch):
        """sum_j P_j * L_avg(delta_i - off_ij); delta (..., 4) -> (..., 4, 4) summed over j"""
        m = np.where(squelch, 0.0, self.p.mod_shift)
        d = delta[..., :, None] - self.ch_off
        d = (d + self.p.fsr / 2) % self.p.fsr - self.p.fsr / 2      # nearest FSR order
        f1 = np.asarray(frac_ones)[..., :, None] if np.ndim(frac_ones) else frac_ones
        return drop_avg(d, self.hwhm[:, None], m[..., :, None] if np.ndim(m) else m, f1)

    def _absorbed(self, delta, pscale, squelch):
        ls = self._lineshape_sum(delta, 0.5, squelch)          # (4, 4)
        return self.p.abs_frac * (ls * (self.p_laser * pscale)[None, :]).sum(-1) * 1e-3  # W

    def _targets(self, ph, pabs, squelch):
        p = self.p
        heat_self = ph                                         # W into own heater
        x_in = self.cmat @ (heat_self * self.r_self)           # K, crosstalk steady state
        e_in = np.where(squelch, 0.0, p.p_mod) * p.r_e
        return np.array([
            heat_self * p.r_hf,
            heat_self * p.r_hs,
            x_in,
            pabs * p.r_shf,
            pabs * p.r_shs,
            e_in,
        ])

    # ------------------------------------------------------------------------------------
    def run_block(self, u_seq, dist):
        """Advance one control block.

        u_seq: (n_sub, 4) heater commands (fraction of FS, dither included), each held 10 us.
        dist:  disturbance dict evaluated at this block's 1 us sample times.
        Returns PD samples (n, 4) in ADC full-scale units (NaN = missing) and truth signals.
        """
        p = self.p
        n_sub = u_seq.shape[0]
        n = dist["dT"].shape[0]
        k = n // n_sub
        steps = np.arange(1, k + 1) * self.dt
        decay = np.exp(-steps[None, :] / self.taus[:, None])  # (6, k)
        temps = np.empty((n, N_RINGS))
        for s in range(n_sub):
            sl = slice(s * k, (s + 1) * k)
            ph = np.clip(u_seq[s], 0.0, 1.0) * p.p_heater_max * self.eff
            j0 = s * k
            jm = j0 + k // 2
            # semi-implicit self heating: evaluate absorption at the mid-substep detuning predicted
            # from the other (input-driven) paths, so the optical feedback is not delayed by the hold
            tgt0 = self._targets(ph, np.zeros(N_RINGS), dist["squelch"][j0])
            mid = k // 2
            other = tgt0 + (self.state - tgt0) * decay[:, mid - 1][:, None]
            temp_mid = other[[0, 1, 2, 5]].sum(0) + self.state[3] + self.state[4]
            d_mid = self._delta(temp_mid, dist["dT"][jm], dist["laser"][jm])
            pabs = self._absorbed(d_mid, dist["pscale"][j0], dist["squelch"][j0])
            tgt = self._targets(ph, pabs, dist["squelch"][j0])
            # z_n = tgt + (z0 - tgt) e^{-n dt / tau}, for n = 1..k
            traj = tgt[:, None, :] + (self.state - tgt)[:, None, :] * decay[:, :, None]  # (6, k, 4)
            self.state = traj[:, -1, :]
            temps[sl] = traj.sum(0)
            self.last_delta = self._delta(temps[sl][-1], dist["dT"][j0 + k - 1], dist["laser"][j0 + k - 1])
        delta = self.unheated + p.kappa * (temps + dist["dT"]) - dist["laser"]      # (n, 4)

        # monitor photodiodes
        frac = 0.5 + self.sigma_ones * self.rng.standard_normal((n, N_RINGS))
        ls = self._lineshape_sum(delta, frac, dist["squelch"])                    # (n, 4, 4)
        pw = self.p_laser[None, :] * dist["pscale"]                                 # (n, 4) per laser j
        drop_mw = (ls * pw[:, None, :]).sum(-1) * self.depth
        pd = drop_mw / p.pd_full_scale_mw + p.pd_noise_fs * self.rng.standard_normal((n, N_RINGS))
        lsb = 1.0 / 2 ** p.pd_bits
        pd = np.clip(np.round(pd / lsb) * lsb, 0.0, 1.0 - lsb)
        f = dist["pdfault"]
        if f.any():
            pd = np.where(f == 1, np.nan, pd)
            pd = np.where(f == 2, 0.31, pd)
            pd = np.where(f == 3, 1.0 - lsb, pd)
        self.t += n * self.dt
        heater_w = np.clip(u_seq, 0, 1).mean(0) * p.p_heater_max * self.eff
        return pd, {"delta_end": delta[-1], "delta_mean": delta.mean(0), "heater_w": heater_w}
