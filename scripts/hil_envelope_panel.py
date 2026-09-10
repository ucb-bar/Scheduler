#!/usr/bin/env python3
"""One combined flight-envelope panel: success vs control rate, colour = cruise speed, Wilson 95% CI.

Packs the whole 5-speed x 4-rate x 6-seed ablation into a single axes:
  * each (speed, rate) cell = a scatter point at its measured success fraction, coloured by
    cruise speed, with a Wilson score 95% CI error bar (right interval for a 6-trial proportion);
  * faint per-speed lines trace each speed's rate response;
  * the POOLED rate marginal (all speeds, n=30/rate — the statistically strong trend) is overlaid
    as a bold line with a shaded 95% CI band;
  * x ticks flag the two showdown rates (50 Hz = ROS, 100 Hz = XPU-RT).

Importable: draw_envelope(ax, csv, ...) renders the panel into any axes (reused by the showdown
figure). Run directly to render the standalone figure.

Data: results/codesign_feedback/hil_ablation.csv. NOTE: fixed controller gain (moment_scale=0.0055,
tuned ~90 Hz), so the 25 Hz collapse is partly under-authority (a gain artifact), not pure Nyquist.
"""
import argparse, csv, math, os
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root, so this runs from any checkout

C_XPU = "#1f9e5a"; C_ROS = "#e2231a"; INK = "#22242a"
DEFAULT_CSV = _REPO + "/results/codesign_feedback/hil_ablation.csv"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - m), min(1.0, c + m)


def _grid(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    cell = defaultdict(lambda: [0, 0])
    for r in rows:
        sp = round(float(r["cruise_speed"]), 3); hz = round(float(r["eff_cmd_hz"]))
        cell[(sp, hz)][1] += 1
        cell[(sp, hz)][0] += int(r["outcome"] == "success")
    speeds = sorted({s for s, _ in cell})
    rates = sorted({h for _, h in cell})
    return cell, speeds, rates


def draw_envelope(ax, csv_path=DEFAULT_CSV, compact=False, colorbar_ax=None, title=True):
    """Render the combined envelope panel into `ax`. Returns the ScalarMappable (for a colorbar)."""
    cell, speeds, rates = _grid(csv_path)
    ns, nr = len(speeds), len(rates)
    xpos = np.arange(nr)                                          # even categorical slot per rate
    cmap = plt.cm.plasma; norm = Normalize(speeds[0] - 0.05, speeds[-1] + 0.05)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    fs = 1.65 if compact else 1.0                                # font scale; compact embed is downscaled ~4x in the composite, so author BIG
    # --- faint per-speed rate-response lines + per-cell scatter with Wilson CI ------------
    for jj, s in enumerate(speeds):
        col = cmap(norm(s)); dx = (jj - (ns - 1) / 2) * 0.15
        ys, los, his = [], [], []
        for i, h in enumerate(rates):
            k, n = cell[(s, h)]
            p, lo, hi = wilson(k, n)
            ys.append(p); los.append(p - lo); his.append(hi - p)
        ax.errorbar(xpos + dx, ys, yerr=[los, his], marker="o", ms=7 * fs, ls="none",
                    capsize=2.5, elinewidth=1.2, mew=1.1 * fs, mec="white",
                    color=col, ecolor=col, alpha=0.72, zorder=4)
    # --- pooled rate marginal (all speeds) with 95% CI band -------------------------------
    pc, plo, phi = [], [], []
    for h in rates:
        k = sum(cell[(s, h)][0] for s in speeds); n = sum(cell[(s, h)][1] for s in speeds)
        p, lo, hi = wilson(k, n); pc.append(p); plo.append(lo); phi.append(hi)
    ax.fill_between(xpos, plo, phi, color=INK, alpha=0.10, zorder=3, lw=0)
    ax.plot(xpos, pc, "-", color=INK, lw=2.6, zorder=5)
    ax.plot(xpos, pc, "D", color=INK, ms=9 * fs, mfc="white", mew=2.2, zorder=6)
    for i, h in enumerate(rates):                                # k/n on the pooled point
        k = sum(cell[(s, h)][0] for s in speeds); n = sum(cell[(s, h)][1] for s in speeds)
        ax.annotate(f"{k}/{n}", (xpos[i], phi[i]), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=9.5 * fs, weight="bold", color=INK, zorder=7)

    # --- axes cosmetics + STORY framing (control-rate floor + speed envelope) ------------
    ax.set_ylim(-0.05, 1.17); ax.set_xlim(-0.45, nr - 0.55)
    xb = 1.5                                                      # floor boundary (between 33 and 50 Hz)
    ax.axvspan(-0.45, xb, color=C_ROS, alpha=0.06, zorder=0)     # under-rate failure zone
    ax.axvspan(xb, nr - 0.55, color=C_XPU, alpha=0.05, zorder=0)  # feasible band
    ax.axvline(xb, color="#b03018", lw=1.5, ls=(0, (5, 3)), alpha=0.6, zorder=1)
    z1 = "under-rate → crash" if not compact else "under-rate"
    z2 = "feasible — speed-limited" if not compact else "feasible"
    ax.text((-0.45 + xb) / 2, 1.11, z1, color="#b81e14", fontsize=11 * fs, weight="bold",
            ha="center", va="center", zorder=8)
    ax.text((xb + nr - 0.55) / 2, 1.11, z2, color="#137a3e", fontsize=11 * fs, weight="bold",
            ha="center", va="center", zorder=8)
    ax.axhline(1.05, color="#d6d3cd", lw=0.8, zorder=0)
    # the significant cliff (25 -> 50 Hz: +37 pts, p<0.001) — the defensible result
    ax.annotate("control-rate floor\n+37 pts · p<0.001", xy=(xb, 0.30), xytext=(0.52, 0.82),
                fontsize=9.5 * fs, weight="bold", color="#111", ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#111", lw=1.0),
                arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.6, connectionstyle="arc3,rad=-0.2"))
    # XPU-RT operates inside the feasible band (full-size only; omitted in the tiny embed to stay legible)
    if not compact:
        ax.annotate("XPU-RT holds 100 Hz\n(above the floor)", xy=(nr - 1, 0.06), xytext=(nr - 1.5, 0.72),
                    fontsize=9 * fs, weight="bold", color=C_XPU, ha="center", va="center", zorder=9,
                    arrowprops=dict(arrowstyle="-|>", color=C_XPU, lw=1.5))
    for x in xpos[:-1]:
        ax.axvline(x + 0.5, color="#ece9e3", lw=0.8, zorder=0.5)
    ax.grid(axis="y", ls=":", lw=0.6, color="#d4d1cb", zorder=0)
    ax.set_xticks(xpos); ax.set_xticklabels([f"{h:g}" for h in rates], fontsize=13 * fs)
    for t, h in zip(ax.get_xticklabels(), rates):     # mark XPU-RT's rate; no ROS/50 head-to-head (unsupported)
        if h == 100:
            t.set_color(C_XPU); t.set_fontweight("bold")
    ax.set_xlabel("control rate (Hz)", fontsize=14 * fs, labelpad=6)
    ax.set_ylabel("success fraction" if compact else "gate-course success fraction", fontsize=14 * fs)
    ax.tick_params(labelsize=12 * fs)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if title and not compact:                                    # compact embed drops its own title — the caption covers it
        ntot = sum(n for (_k, n) in (cell[c] for c in cell))     # total flights (data-driven, not hardcoded)
        nper = sum(cell[(s, rates[0])][1] for s in speeds)       # pooled n per rate
        # THE GAIN CAVEAT, STATED ON THE FIGURE. Every flight in this grid uses a FIXED
        # moment_scale = 0.0055, and that value is calibrated for a 50 Hz closed loop
        # (scripts/hil_dense_grid.sh:25-29). At 25 Hz the correct calibrated gain is 0.02, so
        # the low-rate cells fly UNDER-AUTHORITY by ~4x and part of the apparent rate effect
        # is a gain artifact rather than a rate effect. The gain-corrected grid that would
        # separate them (results/codesign_feedback/hil_dense_grid/) returns 0/8 at EVERY rate
        # including 100-165 Hz, so it settles nothing either. Say so rather than let a reader
        # assume the separation has been done.
        ax.set_title("Flight envelope — success rises with control rate, then speed sets the limit\n"
                     f"{ntot} flights · colour = cruise speed · Wilson 95% CI · black = pooled over speed "
                     f"(n={nper}/rate) · fixed gain (moment_scale 0.0055, calibrated at 50 Hz): "
                     f"low-rate cells are under-authority, so rate and gain are not separated here",
                     fontsize=12.5 * fs, weight="bold", loc="left")
    # label the pooled trend inline near its first point (full-size only)
    if not compact:
        ax.annotate("pooled", (xpos[0], pc[0]), textcoords="offset points", xytext=(9, 11),
                    fontsize=9 * fs, weight="bold", color=INK, va="center", ha="left")

    if colorbar_ax is not None:
        cb = colorbar_ax.figure.colorbar(sm, cax=colorbar_ax)
        cb.set_label("cruise speed (m/s)", fontsize=12 * fs)
        cb.set_ticks(speeds); cb.ax.tick_params(labelsize=10 * fs)
    return sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=_REPO + "/results/codesign_feedback/hil_envelope_combined")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 0.9,
        "xtick.color": INK, "ytick.color": INK})
    fig = plt.figure(figsize=(8.6, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[26, 1], left=0.10, right=0.90, top=0.88, bottom=0.135, wspace=0.04)
    ax = fig.add_subplot(gs[0]); cax = fig.add_subplot(gs[1])
    draw_envelope(ax, a.csv, colorbar_ax=cax)
    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf", "@dpi", a.dpi)


if __name__ == "__main__":
    main()
