# Microring Modulator Thermal Control

This repository holds the literature review and a Python simulation of the thermal locking
controller (TLC) for a four-ring microring-modulator WDM transmitter. The controller combines
dither lock-in sensing, an unscented Kalman filter and a model-predictive controller. It runs in
closed loop against a coupled thermal–optical plant model.

## Layout

```
LitReview/               literature review
docs/report.md           what was implemented, the plant model, results and assumptions
thermal_control/
  tlc/                   the controller, one module per proposal block
    b1_dither.py           B1  dither tones
    b2_conditioner.py      B2  FIR, decimation, validity and squelch flags
    b3_lockin.py           B3  lock-in demodulator
    b4_ukf.py              B4  unscented Kalman filter
    b4_thermal_model.py    B4/B7 reduced-order thermal model
    b5_supervisor.py       B5  mode state machine, loss of lock, fallback
    b6_setpoints.py        B6  operating-point targets
    b7_mpc.py              B7  MPC (fast gradient method)
    b7_pi.py               B7  PI fallback (also the baseline)
    b8_output.py           B8  output stage, slew limit, channel guard
    b9_diagnostics.py      B9  status log and alarms
    b10_calibration.py     B10 calibration store and online learning
    controller.py          TLC top level and BaselinePI
    common.py              ring count and supervisor modes
  plant.py               plant model: thermal paths, self-heating, crosstalk, monitor PD
  disturbances.py        disturbances D1–D6 and fault injection
  params.py              all plant and controller constants
  sim.py, scenarios.py   closed-loop simulation and verification scenarios
  ablation.py            A/B ablation of the TLC blocks
  dither_shapes.py       sine / square / three-level / five-level dither comparison
  advanced_plant.py      controller on the extended plant model
  noise_dither.py        sine vs square dither and PD noise sensitivity
  plots.py               report figures
  results/               figures and metrics.json from the last full run
```

## Usage

Run these from `thermal_control/`, with Python 3.10 or newer:

```bash
pip install -r ../requirements.txt
python3 scenarios.py              # all scenarios + 40-trial campaign (~30 min on 4 cores)
python3 scenarios.py d2_common    # a single scenario
python3 plots.py                  # regenerate results/fig_*.png
python3 ablation.py [variant]     # block A/B tests -> results/ablation.json
python3 noise_dither.py           # PD noise sweeps -> results/noise_dither.json
python3 dither_shapes.py          # dither shape comparison -> results/dither_shapes.json
python3 advanced_plant.py         # extended plant model -> results/advanced_plant.json
```

```python
from sim import run
from disturbances import Disturbances, D2Aggressor
o = run("tlc", T=0.3, dist=Disturbances([D2Aggressor(t0=0.05)]))
o["err"]    # true detuning error per ring (GHz) at the 10 kHz control rate
```

Set `ControllerParams(dither_shape=...)` to `"square"`, `"three"`, `"triangle"` or `"qsine"` for an FPGA-style dither, and use `PlantParams.advanced()` for the extended plant.

See [docs/report.md](docs/report.md) for the results and the modelling assumptions, and
[docs/advisor_feedback.md](docs/advisor_feedback.md) for the noise, square-dither and ablation study, and
[docs/dither_and_plant.md](docs/dither_and_plant.md) for the dither shapes and extended plant.

## License

Copyright 2026 The Microring Modulator Thermal Control authors.

Licensed under the [Apache License, Version 2.0](LICENSE).
