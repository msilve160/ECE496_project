"""Sine vs square dither, and sensitivity to the monitor-PD noise sources.

Run:  python3 noise_dither.py      (writes results/noise_dither.json)
"""
import json
import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from ablation import _scen
from params import PlantParams, ControllerParams
from sim import run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "noise_dither.json")
D = PlantParams()


def _cases():
    c = {"sine": (D, ControllerParams()), "square": (D, ControllerParams(dither_shape="square"))}
    for k in (3, 10):
        c[f"noise x{k}"] = (replace(D, pd_noise_scale=k), ControllerParams())
        c[f"noise x{k} square"] = (replace(D, pd_noise_scale=k), ControllerParams(dither_shape="square"))
    for k in (3, 10):
        c[f"offset x{k}"] = (replace(D, pd_offset_fs=tuple(k * x for x in D.pd_offset_fs)), ControllerParams())
    c["no offset"] = (replace(D, pd_offset_fs=(0.0,) * 4), ControllerParams())
    c["no shot"] = (replace(D, pd_shot=False), ControllerParams())
    c["no Johnson"] = (replace(D, pd_johnson=False), ControllerParams())
    c["offset drift 1%/s"] = (replace(D, pd_offset_drift_fs_per_s=(0.01, -0.01, 0.01, -0.01)), ControllerParams())
    return c


SC = ["steady", "d2_local", "d2_common", "power+2.5%", "squelch", "startup"]


def _job(a):
    case, sc = a
    pp, cp = _cases()[case]
    s = dict(_scen()[sc]); t0 = s.pop("t0")
    kw = dict(T=s.pop("T"), dist=s.pop("dist"), seed=21, **s)
    kw["pp"] = kw.get("pp") and replace(pp, unheated_offset=kw["pp"].unheated_offset) or pp
    try:
        o = run("tlc", cp=cp, **kw)
    except Exception as ex:
        return case, sc, {"error": repr(ex)}
    t, e = o["t"], o["err"]; m = t >= t0; ae = np.abs(e[m])
    mean = e[t > t[-1] - 0.05].mean(0)
    return case, sc, {"p999": float(np.percentile(ae, 99.9, axis=0).max()),
                      "rms": float(np.sqrt((e[m] ** 2).mean())), "peak": float(ae.max()),
                      "final": float(np.abs(e[t > t[-1] - 0.02]).max()),
                      "bias": mean.round(2).tolist(), "alarms": len(o["alarms"])}


if __name__ == "__main__":
    res = {}
    with Pool(4) as pool:
        for c, s, m in pool.imap_unordered(_job, [(c, s) for c in _cases() for s in SC]):
            res.setdefault(c, {})[s] = m
            json.dump(res, open(OUT, "w"), indent=1)
            print("done", c, s, flush=True)
