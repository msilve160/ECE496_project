"""Dither waveform comparison: sine, slew-limited square, three-level and five-level sine.

Run:  python3 dither_shapes.py      (writes results/dither_shapes.json)
"""
import json
import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from ablation import _scen
from params import PlantParams, ControllerParams
from sim import run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "dither_shapes.json")
SHAPES = ["sine", "square", "three", "qsine", "triangle"]
NOISE = {"nominal": 1.0, "noise x3": 3.0}
SC = ["steady", "d2_local", "d2_common", "power+2.5%", "squelch", "startup"]


def _job(a):
    shape, nz, sc = a
    s = dict(_scen()[sc]); t0 = s.pop("t0")
    pp = s.pop("pp", PlantParams())
    pp = replace(pp, pd_noise_scale=NOISE[nz])
    kw = dict(T=s.pop("T"), dist=s.pop("dist"), seed=21, pp=pp, **s)
    try:
        o = run("tlc", cp=ControllerParams(dither_shape=shape), **kw)
    except Exception as ex:
        return shape, nz, sc, {"error": repr(ex)}
    t, e = o["t"], o["err"]; m = t >= t0; ae = np.abs(e[m])
    c = o["ctl"]
    return shape, nz, sc, {"p999": float(np.percentile(ae, 99.9, axis=0).max()),
                           "rms": float(np.sqrt((e[m] ** 2).mean())), "peak": float(ae.max()),
                           "final": float(np.abs(e[t > t[-1] - 0.02]).max()),
                           "bias": e[t > t[-1] - 0.05].mean(0).round(2).tolist(),
                           "fund": c.dither.fund.round(2).tolist(),
                           "cmd_peak_pct": (c.dither.cmd_amp * 100).round(3).tolist(),
                           "slew_hits": int(c.out.slew_hits), "alarms": len(o["alarms"])}


if __name__ == "__main__":
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    import sys
    jobs = [(h, n, s) for h in (sys.argv[1:] or SHAPES) for n in NOISE for s in SC]
    with Pool(4) as pool:
        for h, n, s, m in pool.imap_unordered(_job, jobs):
            res.setdefault(h, {}).setdefault(n, {})[s] = m
            json.dump(res, open(OUT, "w"), indent=1)
            print("done", h, n, s, flush=True)
