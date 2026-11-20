# Advisor feedback round: noise sources, square dither, block ablations

All numbers are worst-ring |detuning error| in GHz, shown as **p99.9 over the window / max over the last 20 ms**. One seed per cell, so differences of a few tenths of a GHz are not significant. Code: `ablation.py`, `noise_dither.py`; raw data in `thermal_control/results/ablation.json` and `noise_dither.json`.

## 1. Plant noise: Johnson-Nyquist, shot, DC offset

The monitor-PD sample now has a signal-dependent noise sigma: shot noise sqrt(2qI·B), Johnson noise sqrt(4kT·B/R_f) (100 kΩ, 300 K, 5 µA full-scale photocurrent), the existing amplifier/ADC term, and a static per-channel DC offset (+0.3, −0.2, +0.4, −0.3 % of full scale) with an optional linear drift. Constants are in `PlantParams` and are assumptions, not datasheet values.

| Case | steady | d2_local | d2_common | power+2.5% | squelch | startup |
|---|---|---|---|---|---|---|
| sine | 0.9 / 0.9 | 2.2 / 0.8 | 2.2 / 0.8 | 2.3 / 1.2 | 3.4 / 1.1 | 1.1 / 1.1 |
| square | 2.3 / 2.3 | 2.7 / 2.3 | 3.0 / 2.2 | 3.6 / 2.6 | 4.5 / 2.3 | 6.5 / 3.0 |
| noise x3 | 1.9 / 1.9 | 2.3 / 1.9 | 3.4 / 1.9 | 3.5 / 2.5 | 3.1 / 1.9 | 2.5 / 2.5 |
| noise x3 square | 1.8 / 1.8 | 2.4 / 1.8 | 3.1 / 1.8 | 3.6 / 2.4 | 3.2 / 1.8 | 6.3 / 3.8 |
| noise x10 | 2.0 / 1.9 | 2.4 / 1.9 | 3.4 / 1.9 | 3.6 / 2.5 | 3.1 / 2.0 | 2.3 / 2.3 |
| noise x10 square | 1.8 / 1.8 | 2.5 / 1.8 | 3.2 / 1.8 | 3.7 / 2.4 | 3.2 / 1.8 | 55.0 / 4.0 |
| offset x3 | 1.3 / 1.3 | 2.3 / 1.3 | 2.5 / 1.2 | 2.7 / 1.7 | 3.8 / 1.5 | 1.4 / 1.4 |
| offset x10 | 2.9 / 2.9 | 3.3 / 2.8 | 4.1 / 2.7 | 4.4 / 3.2 | 5.6 / 3.1 | 227 / 227 |
| no offset | 0.7 / 0.7 | 2.1 / 0.6 | 2.1 / 0.5 | 2.1 / 1.0 | 3.4 / 0.8 | 1.0 / 1.0 |
| no shot | 0.9 / 0.9 | 2.2 / 0.9 | 2.2 / 0.8 | 2.3 / 1.2 | 3.4 / 1.0 | 1.1 / 1.1 |
| no Johnson | 0.9 / 0.9 | 2.2 / 0.8 | 2.2 / 0.8 | 2.3 / 1.2 | 3.4 / 1.0 | 1.1 / 1.1 |
| offset drift 1%/s | 1.1 / 1.1 | 2.2 / 1.0 | 2.3 / 0.9 | 2.3 / 1.4 | 3.4 / 1.1 | 1.3 / 1.3 |

- **Shot and Johnson noise are negligible** at these values. Turning either off changes nothing measurable, because the existing amplifier/ADC floor dominates.
- **Broadband noise ×3 and ×10 raises the steady error from 0.9 to about 1.9 GHz.** It is a bias, not extra jitter: it saturates between ×3 and ×10. Likely cause is the weak second-harmonic anchor (ζ changes only ~0.0015 per GHz), but I did not isolate it.
- **DC offset is the one that matters.** The nominal offsets cost about 0.2 GHz (0.7 without offset, 0.9 with). At ×3 the error is 1.3 GHz and at ×10 (about 3 % FS) it is 2.9 GHz, and a cold start then locks to the wrong channel (227 GHz). Offset drift of 1 % FS/s costs about 0.15 GHz.
- **Not implemented: offset calibration.** Recommended: take a dark reading (laser off or ring far off resonance) at start-up and subtract it per channel.

## 2. Square dither

> Superseded in part: the ~2 GHz bias below came mostly from unmodelled slew-limited edges. See [dither_and_plant.md](dither_and_plant.md) for the fixed model and the other shapes.

An FPGA NCO plus a sign bit gives `sign(cos)` at the 10 µs output rate. Implemented in B1 with these consequences:

- The fundamental of a unit square wave is 4/π, so the command amplitude is scaled by π/4 to keep the same fundamental detuning amplitude (so the lock-in sees the same signal as the sine). Peak command is 21 % lower than the sine's, rms command about 11 % higher.
- The odd harmonics reach the ring through the heater response, so B3/B4's wobble and lock-in models include harmonics 1, 3, 5, 7, 9 (`dither_shape="square"`).
- Tones must be 100 kS/s ÷ N with N even, or the duty cycle is wrong and a spurious 2nd harmonic appears. The set used is 3846, 4545, 6250, 7143 Hz (N = 26, 22, 16, 14), picked so that harmonics and 2f lines stay at least 500 Hz clear of the demodulated lines and the 10 kHz ripple.

Result (table above, rows `sine` vs `square`):

| | sine | square |
|---|---|---|
| steady error | 0.9 | 2.3 (steady bias +2.0, +1.5, +1.5, −0.4 GHz) |
| D2 aggressor | 2.2 | 2.7–3.0 |
| ±2.5 % power + D2 | 2.3 | 3.6 |
| squelch | 3.5 | 4.5 |
| cold start | 1.1 | 6.5 peak, 3.0 final |

**Square dither works but is about 2 GHz worse than sine in my implementation.** It still locks in every scenario at nominal noise and passes a 3 GHz class budget, but the residual ~2 GHz steady bias is not explained. It is not crosstalk, offset or noise (the bias does not shrink without them). My best guess is an error in the modelled harmonic phase or delay (the 16-sample tone's edge lands exactly on a sample, ring 3's second-harmonic reading is off by about −10e-5 against ~0 predicted). With ×3 noise, sine and square are equal (1.9 vs 1.8), so the gap is a model-mismatch term, not a fundamental limit. At ×10 noise, square loses the cold start (55 GHz).
Recommendation: it is viable for an FPGA, but budget the ~2 GHz and expect to re-tune the harmonic model (or fit a per-tone phase on hardware) before relying on it.

## 3. Block A/B ablations (sine dither)

Each row switches off one block or feature; everything else is unchanged.

| Variant (what is removed) | steady | d2_local | d2_common | power+2.5% | squelch | kick_10K | startup |
|---|---|---|---|---|---|---|---|
| **full**: complete controller | 0.9 / 0.9 | 2.2 / 0.8 | 2.2 / 0.8 | 2.3 / 1.2 | 3.4 / 1.1 | 136 / 0.7 | 1.1 / 1.1 |
| **no_mpc**: B7: PI on the lock-in slope instead of the MPC | 0.9 / 0.9 | 157 / 2.2 | 212 / 18.4 | 210 / 1.3 | 5.4 / 0.8 | 142 / 0.6 | 1.3 / 1.3 |
| **no_xtalk**: B4/B7: thermal-crosstalk terms removed from the model | 0.9 / 0.9 | 2.0 / 0.9 | 1.9 / 0.9 | 2.1 / 1.0 | 126 / 1.7 | 136 / 0.7 | 1.2 / 1.2 |
| **no_dc**: B4: fast DC-level channel removed (lock-in ratio only) | 340 / 328 | 342 / 253 | 367 / 253 | 464 / 465 | 385 / 384 | 423 / 381 | 429 / 426 |
| **no_wobble**: B4: dither wobble not modelled in the DC channel | 14.7 / 14.7 | 14.8 / 14.7 | 16.0 / 14.7 | 16.1 / 15.2 | 15.8 / 14.7 | 142 / 14.7 | 12.2 / 12.1 |
| **no_lag**: B4: window-lag compensation of the lock-in measurement | 179 / 83.0 | 10.2 / 8.1 | 11.4 / 8.6 | 10.5 / 7.5 | 129 / 5.8 | 184 / 184 | 11.3 / 12.1 |
| **no_gate**: B4: robust innovation gate | 0.9 / 0.9 | 1.9 / 0.8 | 1.7 / 0.8 | 2.0 / 1.2 | 3.2 / 0.9 | 132 / 1.1 | 1.1 / 1.1 |
| **no_qadapt**: B4: adaptive drift-rate noise | 0.9 / 0.9 | 2.2 / 0.9 | 2.5 / 0.8 | 2.7 / 1.3 | 3.4 / 1.6 | 136 / 0.6 | 1.1 / 1.1 |
| **no_hann**: B3: rectangular instead of Hann lock-in window | 0.9 / 0.9 | 2.2 / 0.8 | 2.2 / 0.7 | 2.2 / 1.2 | 3.4 / 1.0 | 136 / 0.7 | 2.2 / 2.2 |
| **no_phasecal**: B10: online dither-phase calibration | 3.2 / 3.2 | 3.4 / 3.4 | 4.3 / 3.6 | 3.2 / 2.7 | 3.6 / 2.3 | 135 / 2.8 | 3.6 / 3.6 |
| **no_learn**: B10: RLS heater-efficiency and coupling learning | 1.0 / 1.0 | 2.1 / 1.0 | 2.2 / 0.9 | 2.3 / 1.2 | 3.4 / 1.1 | 136 / 0.9 | 1.1 / 1.1 |
| **no_squelch**: B5: squelch recognition | 0.9 / 0.9 | 2.2 / 0.8 | 2.2 / 0.8 | 2.3 / 1.2 | 5.5 / 3.3 | 136 / 0.7 | 1.1 / 1.1 |
| **no_sup**: B5: supervisor (hold, loss of lock, re-acquire) | 0.9 / 0.9 | 2.2 / 0.8 | 2.2 / 0.8 | 2.3 / 1.2 | 3.4 / 1.1 | 136 / 0.7 | 407 / 407 |
| **no_sched**: B1: dither amplitude schedule (fixed 0.3 GHz) | 0.8 / 0.7 | 2.2 / 0.7 | 2.2 / 0.5 | 2.2 / 1.1 | 125 / 2.9 | 136 / 0.3 | 1.4 / 1.4 |
| **baseline**: PI baseline controller (not the TLC) | 1.2 / 1.2 | 49.0 / 1.2 | 530 / 493 | 409 / 409 | 5.3 / 1.1 | 401 / 400 | 400 / 400 |

`kick_10K` applies a deliberate 150 GHz thermal kick, so its p99.9 column is the kick itself; read the "final" column (recovered or not). `no_sup` cannot cold-start (no Acquire).

What each block buys:

- **B4 fast DC channel (`no_dc`)**: essential. Without it the filter loses its scale anchor and every scenario fails (>250 GHz).
- **B4 wobble model (`no_wobble`)**: essential. Without it there is a constant 14.7 GHz bias.
- **B4 lag compensation (`no_lag`)**: essential. 10 GHz on disturbances, and unstable in steady state.
- **B7 MPC (`no_mpc`, PI instead)**: equal at steady state, but PI loses lock under D2 (157–212 GHz) and ±2.5 % power. The TLC's main benefit is dynamic.
- **B10 phase calibration (`no_phasecal`)**: costs about 2–3 GHz of steady bias everywhere (0.9 → 3.2).
- **B5 supervisor (`no_sup`)**: no effect on locked-start runs, but cold start fails (407 GHz).
- **B5 squelch (`no_squelch`)**: squelch final error 3.3 vs 1.1 GHz.
- **B3 Hann window (`no_hann`)**: only the cold start is worse (2.2 vs 1.1).
- **B1 amplitude schedule (`no_sched`)**: slightly *better* steady (0.75), but a 125 GHz transient in the squelch test (final 2.9). One seed, so treat as inconclusive.
- **B4 thermal-crosstalk model (`no_xtalk`)**: no change except one 126 GHz transient in the squelch test, final error 1.7. Inconclusive.
- **B10 RLS learning (`no_learn`)**: no change at nominal plant parameters. It matters only when the plant differs from the model (the `robust` scenarios, not rerun here).
- **B4 robust gate and adaptive drift noise (`no_gate`, `no_qadapt`)**: no measurable benefit in these scenarios (no_gate is marginally better). The test set does not contain the glitches these were added for, so I would keep them but cannot justify them from this data.

Caveats: single seed, short runs, and each ablation is made on top of defaults tuned with the full controller, so a removed block gets no re-tuning.
