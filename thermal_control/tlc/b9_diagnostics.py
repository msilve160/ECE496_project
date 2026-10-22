"""B9 Diagnostics."""
from collections import deque


class B9Diagnostics:
    def __init__(self, rate_hz=100, history_s=10.0):
        self.buf = deque(maxlen=int(rate_hz * history_s))
        self.alarms = []
        self.snapshots = []

    def status(self, t, modes, err, u):
        self.buf.append((t, modes.copy(), err.copy(), u.copy()))

    def alarm(self, t, ring, kind):
        self.alarms.append((t, ring, kind))
        self.snapshots.append(list(self.buf))
