"""Verification scenarios (proposal App. C.1, shortened for simulation time) and their metrics.

Run:  python3 scenarios.py [name ...]      (no names = all; results go to results/)
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from disturbances import (Disturbances, D1Drift, D2Aggressor, D3Uneven, D4Laser, D5Power,
                          D6Squelch, TempKick, PDFault)
from params import PlantParams
from sim import run
from tlc import Mode

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
N = 4


# ---------------------------------------------------------------------------------------
def _ev_force_fault(ctl):
    ctl.force_fault()


def _ev_restart_ring2(ctl):
    ctl.u_cmd[2] = 0.0
    ctl.start_acquire(2)


def metrics(o, t_from=0.0, t_to=None):
    t, e = o["t"], o["err"]
    m = (t >= t_from) & (t <= (t_to if t_to is not None else t[-1] + 1))
    ae = np.abs(e[m])
    ctl = o["ctl"]
    return {
        "p999": np.percentile(ae, 99.9, axis=0).round(3).tolist(),
        "peak": ae.max(0).round(3).tolist(),
        "rms": np.sqrt((e[m] ** 2).mean(0)).round(3).tolist(),
        "heater_mw": (o["heater_w"][m].mean(0) * 1e3).round(3).tolist(),
        "max_slew_pct_per_ms": round(ctl.out.max_step * 100 / (1e3 / ctl.cp.out_rate), 3),
        "cmd_range": [round(ctl.out.out_min, 4), round(ctl.out.out_max, 4)],
        "t_comp_mean_us": round(float(o["t_comp"].mean() * 1e6), 1),
        "t_comp_p99_us": round(float(np.percentile(o["t_comp"], 99) * 1e6), 1),
        "alarms": [(round(float(a), 4), int(b), c) for a, b, c in o["alarms"]][:30],
    }


def recovery_ms(o, t_end, thr=2.0):
    t, e = o["t"], o["err"]
    out = []
    for i in range(N):
        bad = np.where((np.abs(e[:, i]) > thr) & (t > t_end))[0]
        out.append(round(float((t[bad[-1]] - t_end) * 1e3), 2) if len(bad) else 0.0)
    return out


def save(name, kind, o, extra=None):
    os.makedirs(OUT, exist_ok=True)
    keep = ["t", "err", "delta", "target", "delta_hat", "u", "mode", "heater_w", "dT", "eff",
            "cpl", "theta", "fault", "squelch"]
    np.savez_compressed(os.path.join(OUT, f"{name}_{kind}.npz"),
                        **{k: o[k] for k in keep})


# ---------------------------------------------------------------------------------------
def sc_steady(kind):
    o = run(kind, T=1.0, seed=1)
    save("steady", kind, o)
    return {"R1": metrics(o, 0.1)}


def sc_d1(kind):
    # +/-10 K at 10 K/s (10x the design-basis rate, to fit the simulation budget)
    d = Disturbances([D1Drift([(0, 0), (0.05, 0), (1.05, 10), (1.25, 10), (3.25, -10), (3.45, -10),
                               (4.45, 0), (4.6, 0)])])
    o = run(kind, T=4.6, dist=d, seed=2)
    save("d1", kind, o)
    return {"R2": metrics(o, 0.05)}


def sc_d2_common(kind):
    d2 = D2Aggressor(t0=0.05, n_periods=2)
    o = run(kind, T=0.05 + 0.26 + 0.04, dist=Disturbances([d2]), seed=3)
    save("d2_common", kind, o)
    return {"R3": metrics(o, 0.05), "R4_recovery_ms": recovery_ms(o, d2.end_time())}


def sc_d2_local(kind, ring=1):
    d2 = D2Aggressor(t0=0.05, n_periods=2, ring=ring)
    o = run(kind, T=0.05 + 0.26 + 0.04, dist=Disturbances([d2]), seed=4 + ring)
    save(f"d2_local{ring}", kind, o)
    m = metrics(o, 0.05)
    others = [m["peak"][i] for i in range(N) if i != ring]
    return {"R3": m, "R6_others_peak": others, "R4_recovery_ms": recovery_ms(o, d2.end_time())}


def sc_d3_d2(kind, scale=1.0):
    d3 = D3Uneven([0, 1 * scale, 2 * scale, 3 * scale], t0=0.02, t_ramp=0.05)
    d2 = D2Aggressor(t0=0.1, n_periods=1, amp=3.0 * scale, rise_rate=2200 * scale,
                     fall_rate=2300 * scale)
    o = run(kind, T=0.1 + 0.13 + 0.05, dist=Disturbances([d3, d2]), seed=6)
    tag = "d3_d2" if scale == 1.0 else "r11"
    save(tag, kind, o)
    return {"R3_with_D3": metrics(o, 0.02), "recovery_ms": recovery_ms(o, d2.end_time())}


def sc_laser(kind):
    # R5: cold start with initial laser offsets, then compressed 24 h wavelength drift
    d4 = D4Laser(offset_ghz=(20, -20, 35, -35), drift_ghz=(3.5, -3.5, 3.5, -3.5), t0=0.25, dur=0.2)
    o = run(kind, T=0.5, dist=Disturbances([d4]), start="cold", seed=7)
    save("laser", kind, o)
    return {"R5": metrics(o, 0.25), "lock_times": _lock_times(o)}


def sc_power(kind, db=-3.0, ch=None):
    tag_override = None
    if ch is not None:
        tag_override = f"power_{'m' if ch < 0 else 'p'}{abs(ch) * 100:.1f}pct"
    else:
        ch = 10 ** (db / 10) - 1
    d5 = D5Power(change=ch, t0=0.02, dur=0.03)
    d2 = D2Aggressor(t0=0.08, n_periods=1)
    o = run(kind, T=0.25, dist=Disturbances([d5, d2]), seed=8)
    tag = tag_override or f"power_{'m' if db < 0 else 'p'}{abs(int(db))}dB"
    save(tag, kind, o)
    return {"R7": metrics(o, 0.02)}


def sc_squelch(kind, rings=(0, 1, 2, 3)):
    d6 = D6Squelch(t0=0.05, dur=0.075, rings=rings)
    o = run(kind, T=0.2, dist=Disturbances([d6]), seed=9)
    tag = "squelch_all" if len(rings) == 4 else "squelch_one"
    save(tag, kind, o)
    t = o["t"]
    restart = 0.125
    after = (t > restart) & (t < restart + 0.01)
    return {"R12_during": metrics(o, 0.05, 0.125),
            "R12_recovery_ms": recovery_ms(o, restart),
            "after_restart_peak": np.abs(o["err"][after]).max(0).round(3).tolist()}


def _lock_times(o):
    out = []
    for i in range(N):
        tr = np.where(o["mode"][:, i] == Mode.TRACK)[0]
        if len(tr) == 0:
            out.append(None)
            continue
        # time after which the ring stays within 2 GHz for the rest of the run
        bad = np.where(np.abs(o["err"][:, i]) > 2.0)[0]
        out.append(round(float(o["t"][bad[-1] + 1] * 1e3), 2) if len(bad) and bad[-1] + 1 < len(o["t"]) else 0.0)
    return out


def _first_lock_ms(o):
    """Time of each ring's first 'locked' event (end of Acquire), ms."""
    out = [None] * N
    for t, r, kind in o["alarms"]:
        if kind == "locked" and r >= 0 and out[r] is None:
            out[r] = round(float(t) * 1e3, 2)
    return out


def sc_startup(kind):
    pp = PlantParams(unheated_offset=(-120.0, -200.0, -160.0, -250.0))
    o = run(kind, T=0.3, pp=pp, start="cold", seed=10)
    save("startup", kind, o)
    return {"R8_O4_lock_ms": _first_lock_ms(o), "within_2GHz_for_good_ms": _lock_times(o),
            "after_lock": metrics(o, 0.2)}


def sc_stagger(kind):
    o = run(kind, T=0.25, seed=11, events=[(0.01, _ev_restart_ring2)])
    save("stagger", kind, o)
    m = metrics(o, 0.01)
    return {"R8_lock_ms_ring2": _lock_times(o)[2], "R8_others_peak": [m["peak"][i] for i in (0, 1, 3)],
            "all": m}


def sc_lol(kind):
    k = TempKick(ring=2, t0=0.05, kelvin=10.0, t_rise=20e-6)   # 150 GHz: past the resonance
    o = run(kind, T=0.2, dist=Disturbances([k]), seed=12)
    save("lol", kind, o)
    al = [a for a in o["alarms"] if a[1] == 2]
    det = next((a[0] for a in al if a[2] == "loss_of_lock"), None)
    rel = next((a[0] for a in al if a[2] == "locked" and det is not None and a[0] > det), None)
    return {"R10_detect_ms": None if det is None else round((det - 0.05) * 1e3, 2),
            "R10_relock_ms": None if rel is None else round((rel - 0.05) * 1e3, 2),
            "R10_within_2GHz_ms": recovery_ms(o, 0.05)[2], "others_peak": [metrics(o, 0.05)["peak"][i] for i in (0, 1, 3)],
            "alarms": metrics(o)["alarms"]}


def sc_pdfault(kind):
    f = PDFault(ring=1, t0=0.05, dur=0.1, kind=1)
    o = run(kind, T=0.25, dist=Disturbances([f]), seed=13)
    save("pdfault", kind, o)
    t, u = o["t"], o["u"]
    m = (t >= 0.05) & (t <= 0.15)
    return {"R21_cmd_change_pct": float(np.ptp(u[m, 1]) * 100), "err": metrics(o, 0.05)}


def sc_fallback(kind):
    d = Disturbances([D1Drift([(0, 0), (0.1, 0), (0.6, 2.0)])])
    o = run(kind, T=0.6, dist=d, seed=14, events=[(0.05, _ev_force_fault)])
    save("fallback", kind, o)
    t, u = o["t"], o["u"]
    k0 = np.searchsorted(t, 0.05)
    step = np.abs(u[k0:k0 + 3] - u[k0 - 1]).max() * 100
    return {"R22_step_pct": round(float(step), 3), "err": metrics(o, 0.05)}


def sc_robust(kind, eff=0.85, cpl=1.2, tag="robust"):
    plant = PlantParams(heater_eff_scale=(eff,) * 4, coupling_scale=cpl)
    d2a = D2Aggressor(t0=0.05, n_periods=1, ring=1)
    d2b = D2Aggressor(t0=1.6, n_periods=1, ring=1)
    o = run(kind, T=1.8, dist=Disturbances([d2a, d2b]), plant_pp=plant, seed=15)
    save(tag, kind, o)
    return {"early_D2": metrics(o, 0.05, 0.2), "late_D2": metrics(o, 1.6, 1.8),
            "eff_learned": o["eff"][-1].round(3).tolist(), "coupling_learned": round(float(o["cpl"][-1]), 3),
            "truth": {"eff": eff, "coupling": cpl}}


def _trial(seed):
    rng = np.random.default_rng(1000 + seed)
    off = tuple(rng.uniform(-260, -110, 4))
    laser = tuple(rng.uniform(-20, 20, 4))
    amb = rng.uniform(-3, 3)
    pp = PlantParams(unheated_offset=off)
    d = Disturbances([D4Laser(offset_ghz=laser), D1Drift([(0, amb), (1, amb)])])
    o = run("tlc", T=0.3, pp=pp, dist=d, start="cold", seed=seed)
    fin = o["delta"][-200:].mean(0)
    tgt = o["target"][-1]
    wrong = bool(np.any(np.abs(fin - tgt) > 100))
    return {"seed": seed, "lock_ms": _first_lock_ms(o), "settle_ms": _lock_times(o), "final_err": (fin - tgt).round(2).tolist(),
            "wrong_channel": wrong, "max_excursion": float(np.abs(o["delta"] - tgt).max())}


def sc_campaign(n=40, pool=None):
    res = (pool.map(_trial, range(n)) if pool else [_trial(s) for s in range(n)])
    lt = np.array([[x if x is not None else np.nan for x in r["lock_ms"]] for r in res])
    return {"trials": n, "wrong_channel": int(sum(r["wrong_channel"] for r in res)),
            "lock_ms_median": float(np.nanmedian(lt)), "lock_ms_max": float(np.nanmax(lt)),
            "unlocked": int(np.isnan(lt).sum()),
            "final_err_max": float(max(np.abs(r["final_err"]).max() for r in res)), "details": res}


SCENARIOS = {
    "steady": (sc_steady, ["tlc", "baseline"]),
    "d1": (sc_d1, ["tlc", "baseline"]),
    "d2_common": (sc_d2_common, ["tlc", "baseline"]),
    "d2_local1": (lambda k: sc_d2_local(k, 1), ["tlc", "baseline"]),
    "d2_local0": (lambda k: sc_d2_local(k, 0), ["tlc"]),
    "d3_d2": (lambda k: sc_d3_d2(k, 1.0), ["tlc", "baseline"]),
    "r11": (lambda k: sc_d3_d2(k, 1.5), ["tlc"]),
    "laser": (sc_laser, ["tlc"]),
    "power_m3dB": (lambda k: sc_power(k, -3.0), ["tlc"]),
    "power_p3dB": (lambda k: sc_power(k, 3.0), ["tlc"]),
    "power_m2.5pct": (lambda k: sc_power(k, ch=-0.025), ["tlc"]),
    "power_p2.5pct": (lambda k: sc_power(k, ch=0.025), ["tlc"]),
    "squelch_all": (sc_squelch, ["tlc", "baseline"]),
    "squelch_one": (lambda k: sc_squelch(k, (1,)), ["tlc"]),
    "startup": (sc_startup, ["tlc"]),
    "stagger": (sc_stagger, ["tlc"]),
    "lol": (sc_lol, ["tlc"]),
    "pdfault": (sc_pdfault, ["tlc"]),
    "fallback": (sc_fallback, ["tlc"]),
    "robust": (sc_robust, ["tlc"]),
    "robust_hi": (lambda k: sc_robust(k, 1.15, 0.8, "robust_hi"), ["tlc"]),
}


def _job(args):
    name, kind = args
    fn, _ = SCENARIOS[name]
    try:
        return name, kind, fn(kind)
    except Exception as ex:      # keep the campaign going, report the failure
        import traceback
        return name, kind, {"error": repr(ex), "trace": traceback.format_exc()}


def main(names):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(n, k) for n in names if n in SCENARIOS for k in SCENARIOS[n][1]]
    # longest first for better packing
    order = {"d1": 0, "robust": 1, "robust_hi": 1, "steady": 2}
    jobs.sort(key=lambda j: order.get(j[0], 5))
    path = os.path.join(OUT, "metrics.json")
    allm = json.load(open(path)) if os.path.exists(path) else {}
    with Pool(4) as pool:
        for name, kind, m in pool.imap_unordered(_job, jobs):
            allm.setdefault(name, {})[kind] = m
            json.dump(allm, open(path, "w"), indent=1, default=float)
            print("done", name, kind, flush=True)
        if "campaign" in names:
            allm["campaign"] = {"tlc": sc_campaign(40, pool)}
            json.dump(allm, open(path, "w"), indent=1, default=float)
            print("done campaign", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SCENARIOS) + ["campaign"])
