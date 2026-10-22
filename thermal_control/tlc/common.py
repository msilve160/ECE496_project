"""Shared definitions: ring count and the per-ring supervisor modes (B5)."""
from enum import IntEnum

from params import N_RINGS

N = N_RINGS


class Mode(IntEnum):
    INIT = 0
    ACQUIRE = 1
    TRACK = 2
    HOLD = 3
    REACQUIRE = 4
    FAULT = 5
