"""Robustness to the extended plant model (extra thermal poles, heater curvature, thermo-optic
coefficient drift, laser RIN, ADC INL), one effect at a time and all together.

Run:  python3 advanced_plant.py      (writes results/advanced_plant.json)
"""
import json
import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from ablation import _scen
from params import PlantParams, ControllerParams
from sim import run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "advanced_plant.json")
ADV = PlantParams.advanced()
CASES = {
    "nominal plant":       dict(plant={}, shape="sine"),
    "all effects":         dict(plant="adv", shape="sine"),
    "all, three-level":    dict(plant="adv", shape="three"),
    "all, learning off":   dict(plant="adv", shape="sine", learn=False),
    "all, PI baseline":    dict(plant="adv", shape="sine", kind="baseline"),
    "extra heater poles":  dict(plant=dict(r_hm=ADV.r_hm, r_hd=ADV.r_hd), shape="sine"),
    "heater curvature":    dict(plant=dict(heater_curve=ADV.heater_curve), shape="sine"),
    "kappa(T)":            dict(plant=dict(kappa_tc=ADV.kappa_tc), shape="sine"),
    "laser RIN":           dict(plant=dict(rin_dbc_hz=ADV.rin_dbc_hz), shape="sine"),
    "ADC INL":             dict(plant=dict(adc_inl_lsb=ADV.adc_inl_lsb), shape="sine"),
}
SC = ["steady", "d2_local", "d2_common", "power+2.5%", "squelch", "startup"]


def _job(a):
    case, sc = a
    c = CASES[case]
    s = dict(_scen()[sc]); t0 = s.pop("t0")
    base = s.pop("pp", PlantParams())
    pp = ADV if c["plant"] == "adv" else base
    if c["plant"] == "adv":
        pp = replace(ADV, unheated_offset=base.unheated_offset)
    else:
        pp = replace(base, **c["plant"])
    kw = dict(T=s.pop("T"), dist=s.pop("dist"), seed=21, pp=pp, **s)
    try:
        o = run(c.get("kind", "tlc"), cp=ControllerParams(dither_shape=c["shape"]),
                learn=c.get("learn", True), **kw)
    except Exception as ex:
        return case, sc, {"error": repr(ex)}
    t, e = o["t"], o["err"]; m = t >= t0; ae = np.abs(e[m])
    return case, sc, {"p999": float(np.percentile(ae, 99.9, axis=0).max()),
                      "rms": float(np.sqrt((e[m] ** 2).mean())), "peak": float(ae.max()),
                      "final": float(np.abs(e[t > t[-1] - 0.02]).max()),
                      "bias": e[t > t[-1] - 0.05].mean(0).round(2).tolist(),
                      "eff": o["eff"][-1].round(3).tolist(), "alarms": len(o["alarms"])}


if __name__ == "__main__":
    res = {}
    with Pool(4) as pool:
        for c, s, m in pool.imap_unordered(_job, [(c, s) for c in CASES for s in SC]):
            res.setdefault(c, {})[s] = m
            json.dump(res, open(OUT, "w"), indent=1)
            print("done", c, s, flush=True)
