"""Closed-loop co-simulation of the plant with a controller (TLC or baseline)."""
import time
import numpy as np

from disturbances import Disturbances
from params import PlantParams, ControllerParams, N_RINGS
from plant import Plant
from tlc import TLC, BaselinePI

N = N_RINGS


def lock_operating_point(plant: Plant, targets, dist0, iters=60):
    """Heater commands that put every resonance on its target at t = 0 (warm start)."""
    u = np.full(N, 0.3)
    g = plant.p.kappa * plant.p.p_heater_max * (plant.p.r_hf + plant.p.r_hs) * 0.7
    for _ in range(iters):
        plant.settle(u, dist0, n_iter=40)
        err = plant.last_delta - targets
        u = np.clip(u - 0.8 * err / g, 0, 1)
    plant.settle(u, dist0, n_iter=200)
    return u, plant.last_delta.copy()


def run(kind="tlc", T=0.2, dist=None, pp=None, cp=None, seed=0, start="locked",
        events=None, mpc=True, learn=True, verbose=False, plant_pp=None):
    """Simulate for T seconds.

    kind: 'tlc' or 'baseline'. start: 'locked' (warm start at the operating point) or
    'cold' (heaters off, Acquire from the short-wavelength side).
    pp: the controller's nominal device parameters; plant_pp: the actual plant (defaults to pp).
    events: list of (time, callable(controller)) hooks, e.g. forcing the fallback.
    """
    pp = pp or PlantParams()
    cp = cp or ControllerParams()
    plant_pp = plant_pp or pp
    dist = dist or Disturbances()
    rng = np.random.default_rng(seed)
    plant = Plant(plant_pp, rng)
    ctl = TLC(pp, cp, use_mpc=mpc, learn=learn) if kind == "tlc" else BaselinePI(pp, cp)
    targets = ctl.sp.targets
    d0 = dist.evaluate(np.array([0.0]))
    if start == "locked":
        u0, delta0 = lock_operating_point(plant, targets, d0)
        ctl.warm_start(u0, delta0)
    else:
        plant.settle(np.zeros(N), d0, n_iter=50)
        ctl.cold_start()
    ts = cp.ts
    n_pd = int(round(ts * pp.pd_rate))
    steps = int(round(T / ts))
    seq, _, _ = ctl.out.produce(ctl.u_cmd, 0.0, ctl.dither, np.ones(N))
    events = sorted(events or [], key=lambda e: e[0])
    ev_i = 0
    keys = ["t", "delta", "delta_end", "target", "delta_hat", "u", "mode", "heater_w", "sigma",
            "theta", "amp", "fault", "squelch", "rate_hat", "limit", "eff", "cpl", "dT"]
    log = {k: [] for k in keys}
    t_comp = []
    wall = time.time()
    for k in range(steps):
        t_blk = k * ts
        while ev_i < len(events) and events[ev_i][0] <= t_blk:
            events[ev_i][1](ctl)
            ev_i += 1
        tt = t_blk + np.arange(n_pd) / pp.pd_rate
        d = dist.evaluate(tt)
        pd, truth = plant.run_block(seq, d)
        c0 = time.perf_counter()
        seq = ctl.step(pd, t_blk)
        t_comp.append(time.perf_counter() - c0)
        L = ctl.last
        log["t"].append(t_blk + ts)
        log["delta"].append(truth["delta_mean"])
        log["delta_end"].append(truth["delta_end"])
        log["target"].append(ctl.sp.targets.copy())
        log["delta_hat"].append(L["delta_hat"])
        log["u"].append(L["u"])
        log["mode"].append(L["modes"])
        log["heater_w"].append(truth["heater_w"])
        log["sigma"].append(L.get("sigma", np.zeros(N)))
        log["theta"].append(L.get("theta", np.ones(N)))
        log["amp"].append(L["amp"])
        log["fault"].append(L["fault"])
        log["squelch"].append(L["squelch"])
        log["rate_hat"].append(L.get("rate_hat", np.zeros(N)))
        log["limit"].append(L["limit"])
        log["eff"].append(L.get("eff", np.ones(N)))
        log["cpl"].append(L.get("cpl", 1.0))
        log["dT"].append(d["dT"].mean(0))
    out = {k: np.array(v) for k, v in log.items()}
    out["err"] = out["delta"] - out["target"]
    out["t_comp"] = np.array(t_comp)
    out["wall"] = time.time() - wall
    out["alarms"] = list(ctl.diag.alarms)
    out["ctl"] = ctl
    out["slew_hits"] = ctl.out.slew_hits
    if verbose:
        print(f"{kind}: {T:.3f} s simulated in {out['wall']:.1f} s")
    return out
