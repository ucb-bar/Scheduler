#!/usr/bin/env python3
"""One figure that tells the whole HIL story, three panels left-to-right:

  (a) FLIGHT ENVELOPE  — the full speed x rate ablation (course A, 240 flights): a control-rate floor
      gates success, above it cruise speed sets the limit. Reuses hil_envelope_panel.draw_envelope.
  (b) GENERALIZATION   — the same pooled rate-response on a DIFFERENT gate course (B, 120 flights),
      overlaid on course A. The floor + rise reproduces on gates the policy never saw (stack unchanged).
  (c) THE MECHANISM    — WHY below the floor fails: the starved baseline thrashes. Measured mean
      commanded body-moment and modeled propulsive power, as ratios to XPU-RT (log scale). Honestly
      split into MEASURED (moment, straight from the logged wrench) and MODELED (power, mixer +
      momentum theory) so the two claims are never conflated.

Sources: results/codesign_feedback/{hil_ablation.csv, hil_ablation_courseB.csv, flight_energy.csv}.
Output:  results/codesign_feedback/hil_envelope_story.{png,pdf}  (a NEW file — does not overwrite the
         committed hil_envelope_combined used in the paper).
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hil_envelope_panel import draw_envelope, wilson, C_XPU, C_ROS, INK   # reuse the exact envelope panel
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root, so this runs from any checkout

RES = _REPO + "/results/codesign_feedback"
C_A, C_B = INK, "#7b3fa0"                                     # course A = black, course B = violet


def pooled_rate_response(csv_path):
    """pooled-over-speed k,n per control rate -> sorted rates, and (p, lo, hi) each."""
    cell = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(csv_path)):
        hz = round(float(r["eff_cmd_hz"])); cell[hz][1] += 1; cell[hz][0] += int(r["outcome"] == "success")
    rates = sorted(cell)
    stats = [(cell[h][0], cell[h][1], *wilson(*cell[h])) for h in rates]   # k,n,p,lo,hi
    return rates, stats


# ------------------------------- panel (b): generalization -------------------------------------
def draw_generalization(ax, fs=1.0, compact=False):
    rA, sA = pooled_rate_response(f"{RES}/hil_ablation.csv")
    rB, sB = pooled_rate_response(f"{RES}/hil_ablation_courseB.csv")
    x = np.arange(len(rA))
    xb = 1.5                                                   # floor boundary (between 33 and 50 Hz)
    ax.axvspan(-0.45, xb, color=C_ROS, alpha=0.06, zorder=0)
    ax.axvspan(xb, len(rA) - 0.55, color=C_XPU, alpha=0.05, zorder=0)
    ax.axvline(xb, color="#b03018", lw=1.4, ls=(0, (5, 3)), alpha=0.6, zorder=1)
    labA = "course A" if compact else "course A (figure)"
    labB = "course B" if compact else "course B (unseen gates)"
    for (r, s, col, lab, dx) in [(rA, sA, C_A, labA, -0.04), (rB, sB, C_B, labB, 0.04)]:
        p = np.array([t[2] for t in s]); lo = np.array([t[3] for t in s]); hi = np.array([t[4] for t in s])
        ax.fill_between(x, lo, hi, color=col, alpha=0.12, zorder=2, lw=0)
        ax.plot(x + dx, p, "-o", color=col, lw=2.4 * fs, ms=6.5 * fs, mfc="white", mew=1.8, zorder=5, label=lab)
        if not compact:                                       # per-point k/n clutters the small embed
            for i, t in enumerate(s):
                ax.annotate(f"{t[0]}/{t[1]}", (x[i] + dx, hi[i]), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=8.2 * fs, weight="bold", color=col, zorder=7)
    # p-values recomputed by scripts/audit_showdown_claims.py (two-sided Fisher exact):
    # A 25->50 Hz = +36.7 pts, p = 2.7e-07;  B 25->50 Hz = +23.3 pts, p = 0.042.
    # The B value was previously stated as p=0.02, which no test in the repo produces.
    box = ("floor holds on\nunseen gates\nA +37 / B +23 pts" if compact else
           "floor reproduces on\nunseen gates\nA +37 pts (p<0.001)\nB +23 pts (p=0.042)")
    ytop = 0.66 if compact else 1.20                          # compact: tighten range so the rise fills the panel
    ax.annotate(box, xy=(xb, 0.34), xytext=(0.30, 0.55 if compact else 0.87), fontsize=9 * fs, weight="bold",
                color="#111", ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#111", lw=1.0),
                arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.5, connectionstyle="arc3,rad=-0.2"))
    ax.set_ylim(-0.03, ytop); ax.set_xlim(-0.45, len(rA) - 0.55)
    ax.set_xticks(x); ax.set_xticklabels([f"{h:g}" for h in rA], fontsize=12 * fs)
    ax.set_xlabel("control rate (Hz)", fontsize=13 * fs, labelpad=5)
    ax.set_ylabel("success fraction" if compact else "success fraction (pooled over speed)", fontsize=12 * fs)
    ax.tick_params(labelsize=11 * fs)
    ax.grid(axis="y", ls=":", lw=0.6, color="#d4d1cb", zorder=0)
    ax.legend(loc="upper right" if compact else "upper left", fontsize=9.0 * fs, frameon=False,
              bbox_to_anchor=None if compact else (0.0, 1.0), handlelength=1.4, borderaxespad=0.3)
    ax.set_title("Generalization: unseen gates" if compact else "Generalization: same floor, unseen gates",
                 fontsize=11.5 * fs, weight="bold", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ------------------------------- panel (c): the mechanism (energy) ------------------------------
def draw_mechanism(ax, fs=1.0, compact=False):
    rows = list(csv.DictReader(open(f"{RES}/flight_energy.csv")))
    def agg(pref, key):
        g = [r for r in rows if r["flight"].startswith(pref)]
        if key == "power":  vals = [float(r["energy"]) / float(r["dur_s"]) for r in g]   # duration-fair
        else:               vals = [float(r["mean_absM"]) for r in g]
        return sum(vals) / len(vals)
    conds = [("xpu100", "XPU-RT\n100 Hz", C_XPU), ("ros50", "ROS\n50 Hz", C_ROS), ("ros25", "ROS\n25 Hz", "#9a1610")]
    momR = {c: agg(c, "mom") for c, _, _ in conds}; pwR = {c: agg(c, "power") for c, _, _ in conds}
    base_m, base_p = momR["xpu100"], pwR["xpu100"]
    x = np.arange(len(conds)); w = 0.36
    for i, (c, lab, col) in enumerate(conds):
        rm, rp = momR[c] / base_m, pwR[c] / base_p
        ax.bar(x[i] - w / 2, rm, w, color=col, edgecolor=INK, lw=0.8, zorder=3)                       # measured
        ax.bar(x[i] + w / 2, rp, w, color=col, edgecolor=INK, lw=0.8, hatch="////", zorder=3, alpha=0.75)  # modeled
        ax.annotate(f"{rm:.0f}×", (x[i] - w / 2, rm), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9.5 * fs, weight="bold", color=INK, zorder=5)
        ax.annotate(f"{rp:.0f}×", (x[i] + w / 2, rp), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9.5 * fs, weight="bold", color=INK, zorder=5)
    ax.set_yscale("log"); ax.set_ylim(0.6, 900 if compact else 230)   # compact: headroom so the legend clears the bar labels
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=2)
    ax.annotate("XPU-RT = 1×", (len(conds) - 0.5, 1.0), textcoords="offset points", xytext=(0, 3),
                ha="right", fontsize=9 * fs, color=INK, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in conds], fontsize=11 * fs)
    ax.set_ylabel("rel. to XPU-RT (log)" if compact else "relative to XPU-RT (log scale)", fontsize=12 * fs)
    ax.tick_params(labelsize=10 * fs)
    ax.grid(axis="y", ls=":", lw=0.6, color="#d4d1cb", zorder=0)
    leg = [Patch(fc="#888", ec=INK, label="moment (measured)" if compact else "mean commanded moment (measured)"),
           Patch(fc="#888", ec=INK, hatch="////", alpha=0.75, label="power (modeled)" if compact else "propulsive power (modeled)")]
    ax.legend(handles=leg, loc="upper left", fontsize=8.6 * fs, frameon=False)
    ax.set_title("Mechanism: the baseline thrashes", fontsize=11.5 * fs, weight="bold", loc="left")
    if not compact:
        ax.text(0.5, -0.185, "duration-fair (per-second); ratios robust to rotor constants",
                transform=ax.transAxes, ha="center", fontsize=8, color="#666", style="italic")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{RES}/hil_envelope_story")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 0.9,
        "xtick.color": INK, "ytick.color": INK})
    fig = plt.figure(figsize=(17.6, 6.6))
    # envelope gets the most room + its colorbar; generalization and mechanism share the right third
    gs = fig.add_gridspec(1, 4, width_ratios=[26, 1.1, 20, 15], left=0.045, right=0.985,
                          top=0.85, bottom=0.17, wspace=0.32)
    ax_env = fig.add_subplot(gs[0]); cax = fig.add_subplot(gs[1])
    ax_gen = fig.add_subplot(gs[2]); ax_mech = fig.add_subplot(gs[3])
    draw_envelope(ax_env, colorbar_ax=cax, title=False)          # concise title instead of the long 2-line default
    ax_env.set_title("Flight envelope: control-rate floor, then speed-limited",
                     fontsize=11.5, weight="bold", loc="left")
    draw_generalization(ax_gen)
    draw_mechanism(ax_mech)
    for ax, lab in [(ax_env, "a"), (ax_gen, "b"), (ax_mech, "c")]:
        ax.text(-0.02, 1.16, f"({lab})", transform=ax.transAxes, fontsize=15, weight="bold",
                va="top", ha="right", color=INK)
    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf", "@dpi", a.dpi)


if __name__ == "__main__":
    main()
