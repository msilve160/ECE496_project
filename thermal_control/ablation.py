"""A/B ablation of the TLC blocks: switch one block (or feature) off at a time and compare.

Run:  python3 ablation.py [variant ...]      (no names = all; writes results/ablation.json)
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from disturbances import Disturbances, D2Aggressor, D5Power, D6Squelch, TempKick
from params import PlantParams, ControllerParams
from sim import run

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "ablation.json")
N = 4

# variant -> (ablate flags, what is removed)
VARIANTS = {
    "full":        ((), "complete controller"),
    "no_mpc":      (("no_mpc",), "B7: PI on the lock-in slope instead of the MPC"),
    "no_xtalk":    (("no_xtalk",), "B4/B7: thermal-crosstalk terms removed from the model"),
    "no_dc":       (("no_dc",), "B4: fast DC-level channel removed (lock-in ratio only)"),
    "no_wobble":   (("no_wobble",), "B4: dither wobble not modelled in the DC channel"),
    "no_lag":      (("no_lag",), "B4: window-lag compensation of the lock-in measurement"),
    "no_gate":     (("no_gate",), "B4: robust innovation gate"),
    "no_qadapt":   (("no_qadapt",), "B4: adaptive drift-rate noise"),
    "no_hann":     (("no_hann",), "B3: rectangular instead of Hann lock-in window"),
    "no_phasecal": (("no_phasecal",), "B10: online dither-phase calibration"),
    "no_learn":    (("no_learn",), "B10: RLS heater-efficiency and coupling learning"),
    "no_squelch":  (("no_squelch",), "B5: squelch recognition"),
    "no_sup":      (("no_sup",), "B5: supervisor (hold, loss of lock, re-acquire)"),
    "no_sched":    (("no_sched",), "B1: dither amplitude schedule (fixed 0.3 GHz)"),
    "baseline":    (None, "PI baseline controller (not the TLC)"),
}


def _kick(ctl):
    pass


# name -> (T, disturbances, start, metric window, uses supervisor)
def _scen():
    return {
        "steady":    dict(T=0.3, dist=Disturbances(), t0=0.1),
        "d2_local":  dict(T=0.35, dist=Disturbances([D2Aggressor(t0=0.05, n_periods=2, ring=1)]), t0=0.05),
        "d2_common": dict(T=0.35, dist=Disturbances([D2Aggressor(t0=0.05, n_periods=2)]), t0=0.05),
        "power+2.5%": dict(T=0.25, dist=Disturbances([D5Power(change=0.025, t0=0.02, dur=0.03),
                                                       D2Aggressor(t0=0.08, n_periods=1)]), t0=0.02),
        "squelch":   dict(T=0.2, dist=Disturbances([D6Squelch(t0=0.05, dur=0.075)]), t0=0.05),
        "kick_10K":  dict(T=0.2, dist=Disturbances([TempKick(ring=2, t0=0.05, kelvin=10.0, t_rise=20e-6)]), t0=0.05),
        "startup":   dict(T=0.3, dist=Disturbances(), t0=0.2, start="cold",
                          pp=PlantParams(unheated_offset=(-120.0, -200.0, -160.0, -250.0))),
    }


def _job(args):
    var, sc = args
    flags, _ = VARIANTS[var]
    s = dict(_scen()[sc])
    t0 = s.pop("t0")
    kw = dict(T=s.pop("T"), dist=s.pop("dist"), seed=21, **s)
    try:
        o = run("baseline" if flags is None else "tlc", ablate=flags or (), **kw)
    except Exception as ex:
        return var, sc, {"error": repr(ex)}
    t, e = o["t"], o["err"]
    m = t >= t0
    ae = np.abs(e[m])
    tail = np.abs(e[t > t[-1] - 0.02]).max()
    return var, sc, {"p999": float(np.percentile(ae, 99.9, axis=0).max()),
                     "rms": float(np.sqrt((e[m] ** 2).mean())),
                     "peak": float(ae.max()), "final": float(tail),
                     "alarms": len(o["alarms"])}


def main(names):
    names = names or list(VARIANTS)
    jobs = [(v, s) for v in names for s in _scen()]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    with Pool(4) as pool:
        for v, s, m in pool.imap_unordered(_job, jobs):
            res.setdefault(v, {})[s] = m
            json.dump(res, open(OUT, "w"), indent=1)
            print("done", v, s, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
