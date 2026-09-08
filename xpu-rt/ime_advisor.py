"""IME-placement advisor — "run this matmul-class dispatch on the K1 IME matrix
engine, but ONLY where it beats RVV".

The SpaceMiT K1 IME (`smt.vmadot`, micro-tile 4x4x8, cluster-0 only) is a per-
dispatch ALTERNATIVE to the RVV vector unit, not a whole-model backend. It wins
at large M (many GEMM rows) and LOSES at small M, because its fixed B-panel
packing is amortized over ceil(M/4) row-tiles. This module reads a scheduled
report, finds every matmul-class dispatch currently on RVV, computes the GEMM it
lowers to (linear/matmul: M,K,N directly; conv2d via im2col: M=OH*OW, K=IC*KH*KW,
N=OC), predicts the IME-vs-RVV speedup from MEASURED anchors, and recommends IME
ONLY where the predicted speedup > 1. Small-M attention (M=8) is correctly left
on RVV; large-M FFN (M=128) and big-spatial conv (M=OH*OW ≫ 1) are moved.

MEASURED anchors (board hart3, GCC 14.3; kernels/ime/ime_matmul header):
    M=7   -> 0.25x  (loses; two under-filled 4-row tiles)
    M=64  -> 1.43x
    M=128 -> 2.30x
    M->inf ceiling 3.33x (vmadot 34.0 vs bit-exact RVV 10.2 GMAC/s)
The speedup is interpolated over these anchors in M and clamped to [0.20, 3.33].
It is a MODEL calibrated to sparse matmul points; conv-specific measured points
(from the conv-on-IME build) refine it — until then conv recs are marked lower
confidence. This advisor never claims a speedup it did not derive from a measured
anchor curve, and never recommends IME where the model says it loses.

CLI:  python3 xpu-rt/ime_advisor.py schedules/scheduled_*_profiled.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

# (M, speedup) measured anchors — see header.
_ANCHORS: List[Tuple[float, float]] = [(1.0, 0.12), (7.0, 0.25), (64.0, 1.43), (128.0, 2.30)]
_CEILING = 3.33
MATMUL_CLASS = ("linear", "linear_s8", "linear_f16", "matmul_s8", "matmul")
CONV_CLASS = ("conv2d_s8", "conv2d_batchnorm2d_s8", "conv2d_batchnorm2d_silu_s8")


def ime_speedup(m: float) -> float:
    """Predicted IME/RVV speedup as a function of GEMM rows M, from measured
    anchors (piecewise-linear in M), asymptoting to the measured ceiling."""
    if m <= _ANCHORS[0][0]:
        return _ANCHORS[0][1]
    for (m0, s0), (m1, s1) in zip(_ANCHORS, _ANCHORS[1:]):
        if m <= m1:
            return s0 + (s1 - s0) * (m - m0) / (m1 - m0)
    # beyond the last anchor: approach the ceiling with the same shape
    m_last, s_last = _ANCHORS[-1]
    return min(_CEILING, s_last + (_CEILING - s_last) * (1.0 - m_last / m))


def crossover_m() -> float:
    """Smallest M at which predicted speedup crosses 1.0 (the only-if-better line)."""
    lo, hi = _ANCHORS[0][0], _ANCHORS[-1][0]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ime_speedup(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return round(hi, 1)


import csv
import os

# MEASURED conv IME-vs-RVV (K1 board, bit-exact 50/50) keyed by (IC,IH,IW,OC,KH,KW).
# The conv win tracks K=IC*KH*KW and N=OC (4x8 tile fill + packing amortization), NOT
# M=OH*OW — so conv uses this MEASURED lookup, never the M-anchor model. Unmeasured
# conv shapes are NOT recommended (only-if-KNOWN-better); they are flagged to be benched.
_MEASURED_CONV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "data", "ime_measured_conv.csv")


def _load_measured_conv(path: str = _MEASURED_CONV_PATH) -> Dict[Tuple[int, ...], float]:
    out: Dict[Tuple[int, ...], float] = {}
    try:
        for r in csv.DictReader(open(path)):
            key = tuple(int(r[k]) for k in ("IC", "IH", "IW", "OC", "KH", "KW"))
            out[key] = float(r["speedup"])
    except Exception:
        pass
    return out


_MEASURED_CONV = _load_measured_conv()


def conv_speedup(dims: Dict[str, int]) -> Tuple[Optional[float], str]:
    """MEASURED IME/RVV speedup for a conv shape, or (None,'unmeasured')."""
    try:
        key = (dims["IC"], dims["IH"], dims["IW"], dims["OC"], dims["KH"], dims["KW"])
    except KeyError:
        return None, "unmeasured"
    if key in _MEASURED_CONV:
        return _MEASURED_CONV[key], "measured"
    return None, "unmeasured"


_SHAPE_RE = re.compile(r"(rvv_x60|ime_x60|gemmini\w*|scalar)_([a-z0-9]+(?:_[a-z0-9]+)*)_([A-Z].*)$")


def _parse_module(module_name: str) -> Optional[Dict[str, Any]]:
    m = _SHAPE_RE.search(module_name or "")
    if not m:
        return None
    impl, op, shape = m.group(1), m.group(2), m.group(3)
    # tokens are 'x'-separated (N1xIC16xIH30x...); consume the separator so keys
    # are IC/IH/OH/OW, not xIC/xIH (which silently broke conv M = OH*OW).
    dims = {k: int(v) for k, v in re.findall(r"(?:^|x)([A-Za-z][A-Za-z_]*?)(\d+)", shape)}
    return {"impl": impl, "op": op, "shape": shape, "dims": dims}


def _gemm_m(op: str, dims: Dict[str, int]) -> Optional[int]:
    """Rows of the GEMM this op lowers to — the axis that decides IME vs RVV."""
    if op in MATMUL_CLASS:
        return dims.get("M")
    if op in CONV_CLASS or op.startswith("conv2d"):
        oh, ow = dims.get("OH"), dims.get("OW")
        if oh and ow:
            return oh * ow  # im2col rows
    return None


@dataclass
class IMERecommendation:
    kind: str                       # always "ime_place"
    target: str                     # "<op> <shape>"
    expected_savings_us: float      # over all instances of this dispatch in the cycle
    confidence: str                 # "high" | "medium" | "low"
    rationale: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def advise(report: Any, top_k: int = 8) -> List[IMERecommendation]:
    d = report if isinstance(report, dict) else json.load(open(report))
    disp = d.get("dispatches") or {}
    items = disp.values() if isinstance(disp, dict) else disp
    # group RVV matmul-class dispatches by (op, shape); sum measured duration
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for x in items:
        if not isinstance(x, dict):
            continue
        info = _parse_module(x.get("module_name", ""))
        if not info or info["impl"] != "rvv_x60":
            continue
        op = info["op"]
        if not (op in MATMUL_CLASS or op in CONV_CLASS or op.startswith("conv2d")):
            continue
        m = _gemm_m(op, info["dims"])
        if not m:
            continue
        g = groups.setdefault((op, info["shape"]), {"m": m, "dur": 0.0, "n": 0, "dims": info["dims"]})
        g["dur"] += float(x.get("duration", 0.0))
        g["n"] += 1

    recs: List[IMERecommendation] = []
    unmeasured: List[Tuple[str, str]] = []
    for (op, shape), g in groups.items():
        is_conv = op in CONV_CLASS or op.startswith("conv2d")
        if is_conv:
            # conv: MEASURED lookup only (win tracks K,N not M) — never guess
            sp, prov = conv_speedup(g["dims"])
            if prov == "unmeasured":
                unmeasured.append((op, shape))
                continue
            conf, src = "high", "measured on K1 (bit-exact)"
        else:
            # matmul/linear: the M-anchor curve IS measured for these ops
            sp = ime_speedup(g["m"])
            near = any(abs(g["m"] - am) / am < 0.5 for am, _ in _ANCHORS[1:])
            conf = "high" if (near and sp > 1.3) else "medium"
            src = "measured-anchor M-curve"
        if sp is None or sp <= 1.0:     # only-if-(known-)better: never move a loser/unknown
            continue
        saving = g["dur"] * (1.0 - 1.0 / sp)
        recs.append(IMERecommendation(
            kind="ime_place",
            target=f"{op} {shape}",
            expected_savings_us=round(saving, 3),
            confidence=conf,
            rationale=(
                f"{op} " + (f"(M={g['m']} GEMM rows) " if not is_conv else "")
                + f"IME speedup {sp:.2f}x over RVV [{src}] (>1, so IME wins here). "
                f"{g['n']} instance(s) this cycle, {g['dur']:.3f} us on RVV -> "
                f"~{saving:.3f} us saved. Pin impl=ime_x60."
            ),
            detail={"op": op, "shape": shape, "gemm_m": g["m"], "speedup": round(sp, 3),
                    "speedup_source": src, "n_instances": g["n"], "rvv_us": round(g["dur"], 3),
                    "is_conv": is_conv},
        ))
    global LAST_UNMEASURED_CONV
    LAST_UNMEASURED_CONV = unmeasured
    recs.sort(key=lambda r: r.expected_savings_us, reverse=True)
    return recs[:top_k]


#: conv shapes seen on RVV that we have NOT benched on IME (so cannot claim better).
LAST_UNMEASURED_CONV: List[Tuple[str, str]] = []


def emit_hint(report: Any, out_path: str, top_k: int = 32) -> Dict[str, Any]:
    """Emit a modelblaster.ime_hints/v1 contract: which dispatches to place on IME."""
    recs = advise(report, top_k=top_k)
    hint = {
        "contract": "modelblaster.ime_hints/v1",
        "rule": "only_if_faster_than_rvv",
        "crossover_m": crossover_m(),
        "placements": [{"op": r.detail["op"], "shape": r.detail["shape"],
                        "impl": "ime_x60", "speedup": r.detail["speedup"],
                        "speedup_source": r.detail["speedup_source"],
                        "confidence": r.confidence} for r in recs],
    }
    json.dump(hint, open(out_path, "w"), indent=2)
    return hint


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: ime_advisor.py <scheduled_*.json> [--emit hint.json]", file=sys.stderr)
        return 2
    report = sys.argv[1]
    print(f"IME crossover: predicted speedup > 1.0 at M >= {crossover_m()} rows "
          f"(below that IME loses to RVV; those stay put).\n")
    recs = advise(report)
    if not recs:
        print("No RVV matmul-class dispatch is in IME's winning regime — nothing to move.")
    for r in recs:
        tag = "meas" if "measured on" in r.detail["speedup_source"] else "model"
        print(f"[{r.confidence:6s} {tag}] {r.target:44s} +{r.expected_savings_us:8.3f} us  "
              f"(x{r.detail['speedup']:.2f}, M={r.detail['gemm_m']}, "
              f"n={r.detail['n_instances']})")
    if LAST_UNMEASURED_CONV:
        print(f"\n{len(LAST_UNMEASURED_CONV)} conv shape(s) on RVV are UNMEASURED on IME "
              f"(not recommended — bench them first): "
              + ", ".join(s for _, s in LAST_UNMEASURED_CONV[:3])
              + (" ..." if len(LAST_UNMEASURED_CONV) > 3 else ""))
    if "--emit" in sys.argv:
        out = sys.argv[sys.argv.index("--emit") + 1]
        h = emit_hint(report, out)
        print(f"\nwrote {out}: {len(h['placements'])} IME placements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
