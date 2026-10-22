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
    pd_noise_fs: float = 4e-4          # rms TIA/ADC input noise per sample, fraction of full scale (assumed)

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


def coupling_matrix(coupling, n=N_RINGS):
    """Matrix C with C[i, j] = fraction of heater j's DC self-heating seen by ring i (i != j)."""
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            if 0 < d <= len(coupling):
                c[i, j] = coupling[d - 1]
    return c
