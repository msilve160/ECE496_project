"""Figures for the report, from results/*.npz and results/metrics.json.

Run:  python3 plots.py          (after scenarios.py; PNGs go to results/fig_*.png)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optics import drop_avg, lockin_ratios
from params import PlantParams

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RING = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # rings 1-4, fixed order
TLC_C, BASE_C = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SHADE = "#1f2328", "#5b636e", "#e3e6ea", "#eef1f4"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 130, "savefig.dpi": 130,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(name, kind="tlc"):
    p = os.path.join(RES, f"{name}_{kind}.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def ms(t):
    return t * 1e3


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f"fig_{name}.png"), facecolor="white")
    plt.close(fig)
    print("wrote", name)


def rings(ax, t, y, lw=1.4, labels=True):
    for i in range(4):
        ax.plot(t, y[:, i], color=RING[i], lw=lw, label=f"Ring {i + 1}" if labels else None)


def band(ax, lim=2.0):
    ax.axhspan(-lim, lim, color=SHADE, zorder=0, lw=0)


# ---------------------------------------------------------------------------------------------
def fig_operating_point():
    pp = PlantParams()
    h, m = pp.hwhm[0], pp.mod_shift
    d = np.linspace(-60, 60, 600)
    ls = drop_avg(d, h, m)
    z1, _ = lockin_ratios(d, 1.0, h, m)
    from optics import inflection
    op = inflection(h, m, +1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(d, ls, color=RING[0])
    ax[0].axvline(op, color=MUTED, ls="--", lw=1)
    ax[0].annotate(f"operating point\n+{op:.1f} GHz", (op, np.interp(op, d, ls)), (op + 8, 0.75),
                   color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax[0].set(xlabel="Detuning, resonance minus laser (GHz)", ylabel="Drop-port power (norm.)",
              title="Monitor-PD lineshape (data-averaged)")
    ax[1].plot(d, z1 / h, color=RING[0])
    ax[1].axvline(op, color=MUTED, ls="--", lw=1)
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].set(xlabel="Detuning (GHz)", ylabel="z1 / a (1/GHz, norm.)",
              title="First-harmonic lock-in ratio (error signal)")
    save(fig, "operating_point")


def fig_steady():
    a, b = load("steady"), load("steady", "baseline")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for k, (o, name) in enumerate([(a, "TLC (UKF + MPC)"), (b, "Baseline PI")]):
        if o is None:
            continue
        m = o["t"] > 0.1
        band(ax[k], 1.0)
        rings(ax[k], ms(o["t"][m]), o["err"][m], lw=0.8, labels=(k == 0))
        ax[k].set(xlabel="Time (ms)", title=name)
    ax[0].set_ylabel("Detuning error (GHz)")
    ax[0].legend(ncol=4, loc="upper left", fontsize=8)
    save(fig, "steady")


def fig_d2_common():
    a, b = load("d2_common"), load("d2_common", "baseline")
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 2, 2]))
    ax[0].plot(ms(a["t"]), a["dT"][:, 0], color=MUTED)
    ax[0].set(ylabel="dT (K)", title="D2 aggressor at the rings (common to all rings)")
    band(ax[1])
    rings(ax[1], ms(a["t"]), a["err"])
    ax[1].set(ylabel="Error (GHz)", title="TLC: detuning error")
    ax[1].legend(ncol=4, fontsize=8, loc="lower right")
    if b is not None:
        band(ax[2])
        rings(ax[2], ms(b["t"]), np.clip(b["err"], -40, 40), labels=False)
        ax[2].set_ylim(-42, 42)
        ax[2].set(ylabel="Error (GHz, clipped ±40)", title="Baseline PI: detuning error (loses lock)")
    ax[2].set_xlabel("Time (ms)")
    save(fig, "d2_common")


def fig_d2_local():
    a = load("d2_local1")
    b = load("d2_local1", "baseline")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for k, (o, name) in enumerate([(a, "TLC"), (b, "Baseline PI")]):
        if o is None:
            continue
        band(ax[k])
        rings(ax[k], ms(o["t"]), np.clip(o["err"], -40, 40), labels=(k == 0))
        ax[k].set(xlabel="Time (ms)", title=f"{name}: local D2 beside ring 2")
    ax[0].set_ylabel("Detuning error (GHz)")
    ax[0].legend(ncol=2, fontsize=8)
    save(fig, "d2_local")


def fig_d1():
    a, b = load("d1"), load("d1", "baseline")
    fig, ax = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    ax[0].plot(a["t"], a["dT"][:, 0], color=MUTED)
    ax[0].set(ylabel="dT (K)", title="D1 common drift (±10 K at 10 K/s)")
    rings(ax[1], a["t"], a["heater_w"] * 1e3)
    ax[1].set(ylabel="Heater power (mW)", title="TLC heater power")
    ax[1].legend(ncol=4, fontsize=8)
    band(ax[2], 1.0)
    ax[2].plot(a["t"], np.abs(a["err"]).max(1), color=TLC_C, lw=1, label="TLC")
    if b is not None:
        ax[2].plot(b["t"], np.abs(b["err"]).max(1), color=BASE_C, lw=1, label="Baseline PI")
    ax[2].set(ylabel="max |error| (GHz)", xlabel="Time (s)", title="Worst-ring detuning error")
    ax[2].legend(fontsize=8)
    save(fig, "d1")


def fig_startup():
    a = load("startup")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    rings(ax[0], ms(a["t"]), a["delta"])
    ax[0].plot(ms(a["t"]), a["target"][:, 0], color=MUTED, ls="--", lw=1, label="Target (ring 1)")
    ax[0].set(xlabel="Time (ms)", ylabel="Detuning (GHz)", title="Cold start: acquisition ramp")
    ax[0].legend(fontsize=8)
    band(ax[1])
    rings(ax[1], ms(a["t"]), np.clip(a["err"], -10, 10), labels=False)
    ax[1].set(xlabel="Time (ms)", ylabel="Error (GHz, clipped ±10)", title="Settling after lock")
    save(fig, "startup")


def _shade(ax, t0, t1, label):
    ax.axvspan(t0, t1, color=SHADE, lw=0, zorder=0)
    ax.text(t0, 1.0, " " + label, transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=MUTED)


def fig_events():
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.2))
    o = load("squelch_all")
    if o is not None:
        _shade(ax[0, 0], 50, 125, "all rings squelched")
        rings(ax[0, 0], ms(o["t"]), o["err"])
        ax[0, 0].set(title="D6 squelch, all rings (75 ms)", ylabel="Error (GHz)")
        ax[0, 0].legend(ncol=2, fontsize=8, loc="lower right")
    o = load("lol")
    if o is not None:
        _shade(ax[0, 1], 50, 50 + 0.5, "")
        rings(ax[0, 1], ms(o["t"]), np.clip(o["err"], -60, 60), labels=False)
        ax[0, 1].set(title="Loss of lock: +10 K step on ring 3", ylabel="Error (GHz, clipped ±60)",
                     xlim=(40, 120))
    o = load("pdfault")
    if o is not None:
        _shade(ax[1, 0], 50, 150, "ring 2 PD missing: HOLD")
        rings(ax[1, 0], ms(o["t"]), o["err"], labels=False)
        ax[1, 0].set(title="PD fault on ring 2 (100 ms)", ylabel="Error (GHz)", xlabel="Time (ms)")
    o = load("fallback")
    if o is not None:
        ax[1, 1].axvline(50, color=MUTED, lw=1, ls="--")
        ax[1, 1].text(52, 0.97, "switch to PI fallback", transform=ax[1, 1].get_xaxis_transform(),
                      va="top", fontsize=8, color=MUTED)
        rings(ax[1, 1], ms(o["t"]), o["err"], labels=False)
        ax[1, 1].set(title="Forced fallback to PI with drift", ylabel="Error (GHz)", xlabel="Time (ms)")
    save(fig, "events")


def fig_learning():
    o = load("robust")
    o2 = load("robust_hi")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for d, eff_true, ls in [(o, 0.85, "-"), (o2, 1.15, "--")]:
        if d is None:
            continue
        for i in range(4):
            ax[0].plot(d["t"], d["eff"][:, i], color=RING[i], ls=ls, lw=1.2,
                       label=f"Ring {i + 1}" if ls == "-" else None)
        ax[0].axhline(eff_true, color=MUTED, lw=0.8, ls=":")
    ax[0].set(xlabel="Time (s)", ylabel="Heater efficiency estimate",
              title="B10: efficiency (truth 0.85 solid, 1.15 dashed)")
    ax[0].legend(ncol=2, fontsize=8)
    for d, c_true, ls, lab in [(o, 1.2, "-", "truth 1.2"), (o2, 0.8, "--", "truth 0.8")]:
        if d is None:
            continue
        ax[1].plot(d["t"], d["cpl"], color=INK, ls=ls, lw=1.2, label=lab)
        ax[1].axhline(c_true, color=MUTED, lw=0.8, ls=":")
    ax[1].set(xlabel="Time (s)", ylabel="Coupling scale estimate", title="B10: thermal crosstalk scale")
    ax[1].legend(fontsize=8)
    save(fig, "learning")


def fig_summary():
    mpath = os.path.join(RES, "metrics.json")
    if not os.path.exists(mpath):
        return
    M = json.load(open(mpath))
    rows = [("steady", "R1", "Steady state"), ("d1", "R2", "D1 drift"), ("d2_common", "R3", "D2 common"),
            ("d2_local1", "R3", "D2 local"), ("d3_d2", "R3_with_D3", "D3 + D2")]
    labels, tv, bv = [], [], []
    for key, sub, lab in rows:
        if key not in M:
            continue
        g = lambda k: max(M[key][k][sub]["peak"]) if k in M[key] and sub in M[key][k] else np.nan
        labels.append(lab); tv.append(g("tlc")); bv.append(g("baseline"))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(x - 0.2, tv, 0.38, color=TLC_C, label="TLC")
    ax.bar(x + 0.2, bv, 0.38, color=BASE_C, label="Baseline PI")
    for xi, v in zip(x, tv):
        ax.text(xi - 0.2, v * 1.12, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    for xi, v in zip(x, bv):
        if np.isfinite(v):
            ax.text(xi + 0.2, v * 1.12, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set(ylabel="Peak |error|, worst ring (GHz, log)", title="Peak detuning error by scenario")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8)
    save(fig, "summary")


if __name__ == "__main__":
    for f in (fig_operating_point, fig_steady, fig_d2_common, fig_d2_local, fig_d1, fig_startup,
              fig_events, fig_learning, fig_summary):
        try:
            f()
        except Exception as ex:
            print("failed", f.__name__, repr(ex))
