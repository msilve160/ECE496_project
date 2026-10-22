"""Thermal locking controller (TLC): blocks B1-B10 from the proposal (Fig. 1, Table X).

Signal flow per 100 us control step (proposal Section VI-B):
  S1 PD samples -> B2 conditioner -> B3 lock-in -> B4 UKF -> B5 supervisor / B6 setpoints
  -> B7 MPC (or PI fallback) -> B8 output stage (+ B1 dither) -> S2 heater commands
  B9 logs status/alarms; B10 holds calibration and refines heater efficiency and coupling (RLS).

One module per block; `controller.py` wires them into the `TLC` top level.
"""
from .common import Mode, N
from .b1_dither import B1Dither
from .b2_conditioner import B2Conditioner
from .b3_lockin import B3LockIn
from .b4_ukf import B4UKF
from .b4_thermal_model import ThermalModel
from .b5_supervisor import SupervisorMixin
from .b6_setpoints import B6Setpoints
from .b7_mpc import B7MPC
from .b7_pi import PIController
from .b8_output import B8Output
from .b9_diagnostics import B9Diagnostics
from .b10_calibration import B10Calibration
from .controller import TLC, BaselinePI

__all__ = ["Mode", "N", "B1Dither", "B2Conditioner", "B3LockIn", "B4UKF", "ThermalModel",
           "SupervisorMixin", "B6Setpoints", "B7MPC", "PIController", "B8Output", "B9Diagnostics",
           "B10Calibration", "TLC", "BaselinePI"]


