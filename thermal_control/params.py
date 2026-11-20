"""All numerical parameters for the plant model and the thermal locking controller (TLC).

Units used everywhere unless a name says otherwise:
  detuning / frequency offsets: GHz (positive = resonance at a LONGER wavelength than its laser)
  temperature: K, time: s, heater command u: fraction of full scale (0..1), optical power: mW

Values marked "proposal" come from the ECE496 proposal (Tables I, II, X, App. C).
Values marked "ref [4]" come from Cherian et al., SPIE 12891 128910D (Table 1).
Everything else is a stated simulation assumption (see the report).
"""
from dataclasses import dataclass
import numpy as np

N_RINGS = 4


@dataclass
class PlantParams:
    # --- thermo-optic -------------------------------------------------------------
    kappa: float = 15.0                # GHz/K, proposal Table I (86 pm/K at 1310 nm)
    channel_spacing: float = 400.0     # GHz, proposal C2
    fsr: float = 2000.0                # GHz, assumed ring free spectral range (> 4 channels)

    # --- heater (ref [4] Table 1a, linearised: command = power fraction) ----------
    p_heater_max: float = 25e-3        # W full scale (assumed; gives ~580 GHz self tuning)
    r_hf: float = 1135.0               # K/W fast heater path
    tau_hf: float = 13e-6              # s
    r_hs: float = 410.0                # K/W slow heater path
    tau_hs: float = 78e-6              # s
    # thermal crosstalk: fraction of a heater's DC self-heating seen by a ring 1, 2, 3 positions away
    coupling: tuple = (0.20, 0.06, 0.02)   # assumed (proposal: crosstalk exists, values not given)
    tau_x: float = 250e-6              # s, crosstalk path through the substrate (assumed, slower than self)

    # --- optical self heating (ref [4] Table 1b) -----------------------------------
    r_shf: float = 465.0
    tau_shf: float = 1.7e-6
    r_shs: float = 2440.0
    tau_shs: float = 7.7e-6
    abs_frac: float = 0.30             # fraction of laser power absorbed at the resonance peak (assumed)

    # --- extended thermal / electrical effects (all off by default; PlantParams.advanced() turns them on)
    r_hm: float = 0.0                  # K/W extra medium heater path (membrane / oxide spreading)
    tau_hm: float = 400e-6             # s
    r_hd: float = 0.0                  # K/W extra slow heater path (die / package spreading)
    tau_hd: float = 4e-3               # s
    heater_curve: float = 0.0          # beta in P = Pmax (u + beta u (1-u)): heater power not linear in the command
    kappa_tc: float = 0.0              # 1/K, relative rise of the thermo-optic coefficient with ring temperature
    rin_dbc_hz: float = -200.0         # laser relative intensity noise, dBc/Hz (white)
    adc_inl_lsb: float = 0.0           # peak static ADC integral nonlinearity, LSB

    # --- modulator electrical self heating (ref [4] Table 1c) ----------------------
    r_e: float = 2905.0
    tau_e: float = 0.2e-6
    p_mod: float = 0.1e-3              # W average driver dissipation in the ring (assumed)

    # --- ring lineshape (Lorentzian drop response), per ring fabrication spread ----
    hwhm: tuple = (20.4, 19.6, 20.9, 19.3)     # GHz half width (FWHM ~40 GHz, loaded Q ~5.7k)
    depth: tuple = (0.95, 0.93, 0.96, 0.95)    # through-port dip depth at resonance
    mod_shift: float = 8.0                      # GHz, +/- resonance shift from the 53.125 GBd NRZ drive

    # --- lasers ---------------------------------------------------------------------
    laser_power: tuple = (1.00, 0.92, 1.08, 0.97)   # mW per channel (assumed spread)
    sym_rate: float = 53.125e9         # Bd, proposal C2

    # --- monitor photodiode + interface electronics (S1) -----------------------------
    pd_rate: float = 1e6               # S/s, proposal R26
    pd_bits: int = 12                  # proposal R26
    pd_full_scale_mw: float = 1.35     # drop-port optical power giving ADC full scale (assumed)
    # Noise sources of the monitor chain (all assumed). The three physical ones are computed from
    # the photocurrent: the tap puts pd_fs_current_ua on the TIA at ADC full scale.
    pd_fs_current_ua: float = 5.0      # photocurrent at ADC full scale (~0.4 % tap of the drop port)
    pd_rf_kohm: float = 100.0          # TIA feedback resistor: Johnson-Nyquist current noise 4kT/Rf
    pd_temp_k: float = 300.0           # K
    pd_noise_fs: float = 3.8e-4        # rms amplifier + ADC excess noise per sample, fraction of FS
    pd_noise_scale: float = 1.0        # multiplies all three random noise sources (sensitivity sweeps)
    pd_shot: bool = True               # signal-dependent shot noise sqrt(2 q I B)
    pd_johnson: bool = True            # Johnson-Nyquist noise of the TIA feedback resistor
    pd_offset_fs: tuple = (0.003, -0.002, 0.004, -0.003)   # static DC offset per ring, fraction of FS
    pd_offset_drift_fs_per_s: tuple = (0.0, 0.0, 0.0, 0.0)  # slow linear offset drift

    @classmethod
    def advanced(cls, **kw):
        """Plant with the extended effects on: two extra heater poles (about +13 % DC gain the
        controller's two-pole model does not have), a 12 % bow in heater power vs command, a 0.1 %/K
        rise of the thermo-optic coefficient, laser RIN of -130 dBc/Hz, and 1.5 LSB of ADC INL."""
        d = dict(r_hm=150.0, r_hd=60.0, heater_curve=0.12, kappa_tc=1e-3, rin_dbc_hz=-130.0, adc_inl_lsb=1.5)
        d.update(kw)
        return cls(**d)

    def noise_sigma_fs(self, level=None):
        """rms noise per sample (fraction of FS) at signal `level` (fraction of FS; array ok)."""
        q, kb = 1.602e-19, 1.381e-23
        bw = self.pd_rate / 2                       # noise bandwidth: Nyquist of the 1 MS/s sampling
        i_fs = self.pd_fs_current_ua * 1e-6
        var = np.full_like(np.asarray(0.0 if level is None else level, float), self.pd_noise_fs ** 2)
        if self.pd_johnson:
            var = var + (4 * kb * self.pd_temp_k * bw / (self.pd_rf_kohm * 1e3)) / i_fs ** 2
        if self.pd_shot and level is not None:
            var = var + 2 * q * bw * np.maximum(level, 0.0) / i_fs
        return self.pd_noise_scale * np.sqrt(var)

    def noise_floor_fs(self):
        """Noise rms at a typical operating level with no data-pattern noise: what the controller
        calibrates as its squelch floor."""
        return float(self.noise_sigma_fs(0.4))

    # --- unheated resonance positions (relative to each ring's own laser) ------------
    # Default: ~250 GHz short of the operating point so heaters sit mid-range and the D1-D3
    # envelope stays inside heat-only authority (proposal C4).
    unheated_offset: tuple = (-240.0, -255.0, -248.0, -262.0)

    # --- plant/model mismatch knobs (1.0 = plant equals the controller's nominal model) --
    heater_eff_scale: tuple = (1.0, 1.0, 1.0, 1.0)
    coupling_scale: float = 1.0


@dataclass
class ControllerParams:
    ts: float = 100e-6                 # s, 10 kHz update (proposal R25)
    out_rate: float = 100e3            # S/s heater command stream (S2 >= 10 kS/s; 100 kS/s chosen to carry dither)
    decim: int = 2                     # B2 decimation 1 MS/s -> 500 kS/s
    fir_taps: int = 31
    fir_cutoff: float = 150e3          # Hz
    lockin_window: float = 4e-3        # s; tones on a 250 Hz grid are orthogonal over this window
    # B1: tones on the 250 Hz bin grid chosen so every tone and 2nd harmonic is >= 2 bins from
    # every other tone, harmonic and 2nd-order intermodulation product (f_j +/- f_k), from the
    # 10 kHz control-update ripple and >= 8 bins from DC (Hann window in B3). Kept low because the
    # R19 slew limit (10 % FS/ms) caps dither amplitude x frequency.
    tone_freqs: tuple = (2.0e3, 3.0e3, 3.5e3, 4.5e3)
    # Dither waveform: "sine" or "square" (an FPGA toggles a square wave trivially). A square wave puts
    # energy in odd harmonics, so its tones are re-chosen (SQUARE_TONES) to keep every harmonic up to
    # the 9th, the 2nd-order intermodulation products and the 10 kHz ripple >= 500 Hz from every
    # demodulated line. Amplitudes are set so the *fundamental* detuning amplitude equals the sine's.
    dither_shape: str = "sine"        # "sine", "square", "three" (3-level), "triangle" or "qsine" (5-level sine)
    dither_min: float = 0.1            # GHz peak detuning (proposal B1)
    dither_max: float = 0.5
    dither_nominal: float = 0.3

    # B4 UKF
    ukf_q_delta: float = 0.05          # GHz per step, random walk on detuning
    ukf_q_rate_common: float = 600.0   # GHz/s per step, common-mode drift-rate random walk
    ukf_q_rate_grad: float = 300.0     # GHz/s per step, per gradient mode (linear, quadratic)
    ukf_q_rate_local: float = 250.0    # GHz/s per step, independent per ring
    ukf_q_adapt: float = 30.0          # max multiplier of rate noise when innovations are large
    ukf_gate: float = 25.0             # innovation gate (normalised innovation squared)
    ukf_q_p: float = 1e-5              # relative random walk of the PD DC scale per step
    meas_lag_steps: float = 20.0       # lock-in window centre lag in control steps

    # B7 MPC
    horizon: int = 20                  # proposal Table X
    q_weight: float = 1.0              # per GHz^2
    r_weight: float = 2e-2             # per u^2 (heater power)
    s_weight: float = 4e3              # per (du)^2
    slew_per_step: float = 0.006       # fraction of FS per 100 us step (R19 allows 0.01; margin for dither)
    slew_penalty: float = 2e5
    fgm_iters: int = 25
    rate_decay: float = 0.93           # per-step decay of the extrapolated drift rate in the prediction

    # B7 fallback / baseline PI
    rate_max: float = 40e3            # GHz/s, bound on the UKF drift-rate state
    pi_kp: float = 0.10                # GHz of correction per GHz error (scaled by plant DC gain)
    pi_ki: float = 150.0               # 1/s
    pi_tt: float = 2e-3                # s, back-calculation anti-windup time constant

    # B5 supervisor
    acquire_rate: float = 2.0          # GHz/ms nominal heater ramp during Acquire
    reacquire_rate: float = 6.0        # GHz/ms
    reacquire_backoff: float = 120.0   # GHz of heater reduction before re-ramping
    lol_err: float = 25.0              # GHz estimated error that declares loss of lock
    lol_time: float = 1.5e-3           # s persistence before declaring loss of lock
    limit_flag_time: float = 2e-3      # s a heater limit must block correction before R20 flag
    channel_guard: float = 200.0       # GHz, R23 / B8 travel bound

    # B6 setpoints
    oma_shift_max: float = 4.0         # GHz, kept inside W-OP with margin
    er_min_db: float = 3.5             # R13 (enforced over +/-2 GHz; see report)
    er_check_window: float = 2.0

    # B10 calibration
    rls_lambda: float = 0.9995
    eff_bounds: tuple = (0.7, 1.3)
    coupling_bounds: tuple = (0.6, 1.5)

    def __post_init__(self):
        if self.dither_shape != "sine" and tuple(self.tone_freqs) == (2.0e3, 3.0e3, 3.5e3, 4.5e3):
            self.tone_freqs = SQUARE_TONES


# f = out_rate / N with N even (26, 22, 16, 14 samples per period): every half period is a whole number of
# 10 us output samples, so the toggled square wave has no duty-cycle error (and hence no 2nd harmonic).
# The only such set with >= 500 Hz between every demodulated line and every other line.
SQUARE_TONES = tuple(100e3 / n for n in (26, 22, 16, 14))


def coupling_matrix(coupling, n=N_RINGS):
    """Matrix C with C[i, j] = fraction of heater j's DC self-heating seen by ring i (i != j)."""
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            if 0 < d <= len(coupling):
                c[i, j] = coupling[d - 1]
    return c
