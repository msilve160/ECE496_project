# Dither shapes and extended plant model

Follow-up to [advisor_feedback.md](advisor_feedback.md). Error is the worst-ring |detuning| in GHz, shown as **p99.9 over the window / max over the last 20 ms**. One seed per cell. Code: `dither_shapes.py`, `advanced_plant.py`; data in `results/dither_shapes.json` and `results/advanced_plant.json`.

## 1. Dither shapes

**The earlier ~2 GHz square-dither bias was largely a slew-limit effect, and it is now modelled.** The R19 limit is 10 % FS/ms, which is 0.1 % FS per 10 µs output sample. A square wave of the amplitude needed at kHz tones has edges of 0.2–0.3 % FS, so the output stage was silently slewing them while the estimator assumed ideal edges. A square dither under R19 is therefore always a trapezoid, and the higher the tone, the closer it is to a triangle.

What changed in the code (B1/controller):
- B1 builds each tone's one-period pattern with its edges limited to 35 % of the slew budget, and the base command keeps the rest. The pattern shape is fixed for the largest scheduled amplitude, so smaller amplitudes only rescale it.
- The estimator expands the actual pattern into harmonics 1–9 (including the sample-and-hold sinc) instead of assuming the ideal 4/π, 1/k series, and it carries the ramp delay in the fundamental's phase.
- Four shapes are available through `ControllerParams(dither_shape=...)`: `square`, `three` (levels +1/0/−1, thresholds at cos = ±0.5), `triangle` and `qsine` (sine rounded to five levels). Tones are 100 kS/s ÷ N for even N for all of them.

| Dither (nominal noise) | steady | d2_local | d2_common | power+2.5% | squelch | startup |
|---|---|---|---|---|---|---|
| sine | 0.9 / 0.9 | 2.2 / 0.8 | 2.2 / 0.8 | 2.3 / 1.2 | 3.4 / 1.1 | 1.1 / 1.1 |
| square | 2.0 / 2.0 | 2.7 / 1.9 | 3.2 / 1.9 | 3.3 / 2.4 | 4.6 / 2.3 | 3.5 / 3.5 |
| three | 1.4 / 1.3 | 2.0 / 1.3 | 2.5 / 1.2 | 2.7 / 1.7 | 3.8 / 1.5 | 1.4 / 1.4 |
| qsine | 10.7 / 10.5 | 10.1 / 9.9 | 11.3 / 10.4 | 11.8 / 11.3 | 9.0 / 7.9 | 10.7 / 10.4 |
| triangle | 0.6 / 0.4 | 1.9 / 0.4 | 2.1 / 0.5 | 2.1 / 0.7 | 3.4 / 0.5 | 1.4 / 1.3 |

| Dither (noise ×3) | steady | d2_local | d2_common | power+2.5% | squelch | startup |
|---|---|---|---|---|---|---|
| sine | 1.9 / 1.9 | 2.3 / 1.9 | 3.4 / 1.9 | 3.5 / 2.5 | 3.1 / 1.9 | 2.5 / 2.5 |
| square | 1.8 / 1.8 | 2.4 / 1.8 | 3.2 / 1.8 | 3.7 / 2.4 | 3.1 / 1.8 | 6.6 / 6.6 |
| three | 1.8 / 1.8 | 2.4 / 1.8 | 3.2 / 1.8 | 3.7 / 2.4 | 3.0 / 1.8 | 4.2 / 4.2 |
| qsine | 1.8 / 1.8 | 2.4 / 1.8 | 3.2 / 1.8 | 3.7 / 2.4 | 3.0 / 1.8 | 5.8 / 5.8 |
| triangle | 1.8 / 1.8 | 2.4 / 1.8 | 3.2 / 1.8 | 3.7 / 2.4 | 3.1 / 1.8 | 3.6 / 3.6 |

- **Triangle is the best shape tested, including against sine, at nominal noise:** 0.55 GHz steady error (sine 0.91), 1.9 / 2.1 GHz for local / common D2 (sine 2.2 / 2.2), 2.1 for the power step (2.3) and the same squelch result. Cold start is slightly worse than sine (1.4 against 1.1) and at noise ×3 it ties the others in steady state but is worse than sine at start-up (3.6 against 2.5). It is also the shape that makes the best use of the slew budget, as predicted. Its command peak is 0.13–0.30 % FS, about the same as square and three-level.
- **Three-level is the second-best FPGA-friendly shape:** 1.4 GHz in steady state against 0.9 for sine, 2.0 against 2.2 for a local D2 aggressor and 1.4 against 1.1 for cold start. Square is worse (2.0 / 2.7 / 3.5). Both still use roughly the same trapezoid edges, so the gain comes from the lower harmonic content.
- **Five-level sine fails at nominal noise** (about 10 GHz biased, ring 3 at 7.1 kHz). Its edges are too fast for the slew budget at that tone, and there were slew violations. At noise ×3 it behaves like the others. Treat it as unreliable until its tone set is revisited.
- **The cost of a sampled shape is command amplitude.** For the same fundamental detuning amplitude, the heater command peaks at 0.11 % FS (sine) but 0.25–0.29 % FS at rings 3–4 for square and three-level, because slew limiting cuts the fundamental to 0.43–0.89 of the peak. That larger swing is the price of the simpler hardware.
- **Under noise ×3 all shapes converge** to about 1.8 GHz in steady state: noise, not shape, is the limit there.
- **Remaining issues:** some slew violations remain (about 80 of 3 × 10⁴ samples for square) because the base command can add its own slew on top of the dither edges. The remaining 0.5–1 GHz bias against sine is not explained; the next candidate is an error in the modelled phase of the higher harmonics. The triangle result suggests the remaining bias in the other shapes comes from their harmonic content rather than the model.

## 2. Extended plant model

All of these are off by default and are switched on together with `PlantParams.advanced()`. The values are my assumptions (no device data), chosen to stress the controller rather than predict a real device.

| Effect | Implementation |
|---|---|
| Distributed heat spreading | Two extra heater poles (150 K/W at 400 µs and 60 K/W at 4 ms), about +13 % DC gain that the controller's two-pole model does not have |
| Heater nonlinearity | P = Pmax (u + 0.12 u(1−u)): power bows 12 % against the command |
| Thermo-optic coefficient | κ rises 0.1 %/K with ring temperature |
| Laser RIN | −130 dBc/Hz, white, multiplies every channel's power |
| ADC INL | 1.5 LSB peak, static |

| Plant | steady | d2_local | d2_common | power+2.5% | squelch | startup |
|---|---|---|---|---|---|---|
| nominal plant | 0.9 / 0.9 | 2.2 / 0.8 | 2.2 / 0.8 | 2.3 / 1.2 | 3.4 / 1.1 | 1.1 / 1.1 |
| all effects | 1.6 / 1.5 | 3.2 / 1.5 | 3.2 / 1.7 | 2.4 / 1.2 | 3.9 / 1.5 | 2.3 / 2.2 |
| all, three-level | 1.0 / 1.0 | 2.1 / 1.0 | 2.4 / 1.2 | 1.8 / 0.7 | 3.6 / 0.8 | 2.1 / 1.8 |
| all, learning off | 2.1 / 2.1 | 3.3 / 2.1 | 3.5 / 2.2 | 2.8 / 1.9 | 3.9 / 1.9 | 2.4 / 2.4 |
| all, PI baseline | 2.4 / 2.3 | 51.9 / 2.3 | 503.7 / 466.6 | 405.8 / 401.1 | 7.3 / 2.3 | 2.8 / 2.7 |
| extra heater poles | 0.8 / 0.7 | 2.2 / 0.7 | 2.2 / 0.6 | 2.1 / 1.1 | 3.4 / 0.9 | 1.0 / 1.0 |
| heater curvature | 1.0 / 1.0 | 2.9 / 1.1 | 2.9 / 1.1 | 1.8 / 0.6 | 3.5 / 0.9 | 1.9 / 1.8 |
| kappa(T) | 0.7 / 0.7 | 2.5 / 0.7 | 2.5 / 0.7 | 1.9 / 0.7 | 3.6 / 0.9 | 1.4 / 1.3 |
| laser RIN | 0.9 / 0.9 | 2.2 / 0.8 | 2.2 / 0.7 | 2.3 / 1.2 | 3.4 / 1.1 | 1.1 / 1.1 |
| ADC INL | 0.9 / 0.9 | 2.1 / 0.9 | 2.1 / 0.8 | 2.3 / 1.2 | 3.4 / 1.0 | 1.1 / 1.1 |

- **The TLC tolerates each effect on its own and all of them together:** steady error 0.9 → 1.6 GHz, D2 local 2.2 → 3.2 GHz, and no loss of lock.
- **RIN and ADC INL are negligible** at these levels. **Heater curvature and the extra poles are what move the numbers**, mostly through the learned efficiency (B10 settles at 1.00–1.03).
- **Learning matters on the extended plant:** with B10 learning off, steady error is 2.1 against 1.6 GHz.
- **Three-level dither holds up on the extended plant too** (1.0 GHz steady).
- **The PI baseline loses lock** under D2 and the power step, as before.

Caveat: the extended effects are modelled as plant-only; the controller's internal model is unchanged, so this tests robustness to mismatch, not a better model.

## Not done
Lineshape asymmetry (Fano), free-carrier or two-photon absorption, optical crosstalk between rings beyond the Lorentzian tails, laser linewidth, and a finite-element thermal model. The extra poles above are a crude stand-in for the last of these.
