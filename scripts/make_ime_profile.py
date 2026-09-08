#!/usr/bin/env python3
"""Emit an ime_x60 profile CSV for a conv net from its rvv_x60 profile + the
MEASURED conv IME-vs-RVV table, so the scheduler can place conv dispatches on
the K1 matrix engine — and only where IME is measured faster.

For each dispatch in the rvv profile:
  * conv2d* with a MEASURED speedup > 1  -> cycles = round(rvv_cycles/speedup),
    implementation = curated[ime]/ime_vmadot_4x4x8, module_name rvv_x60->ime_x60.
  * everything else (conv losers, non-conv ops) -> copied verbatim from the rvv
    profile (same cost), so IME is never cheaper there and the solver keeps RVV.

The IME cost is the measured SPEEDUP applied to THIS profile's rvv baseline
(not the standalone bench's absolute cycles), so rvv and ime cells are on the
same clock and the per-dispatch min the solver takes is apples-to-apples.

Usage:
  python scripts/make_ime_profile.py --net dronet
  python scripts/make_ime_profile.py --net yolov8_nano
"""
import argparse
import csv
import hashlib
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEASURED = os.path.join(REPO, "xpu-rt", "data", "ime_measured_conv.csv")


def _load_measured():
    out = {}
    for r in csv.DictReader(open(MEASURED)):
        out[tuple(int(r[k]) for k in ("IC", "IH", "IW", "OC", "KH", "KW"))] = float(r["speedup"])
    return out


def _parse_shape(shape: str) -> dict:
    return {k: int(v) for k, v in re.findall(r"([A-Za-z]+)=(\d+)", shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--variant", default="int8",
                    help="quant/variant tag, e.g. int8 | reffused.int8 | unfused.int8")
    ap.add_argument("--target", default="spacemit_x60")
    ap.add_argument("--topo", default="topo_0")
    args = ap.parse_args()

    v = args.variant
    rvv = (f"gen/profile_mb/rvv_x60/{args.target}/{args.net}/{args.net}.{v}/"
           f"{args.net}_{args.target}_rvv_x60_{args.net}.{v}/{args.topo}/results.csv")
    rvv_path = os.path.join(REPO, rvv)
    if not os.path.exists(rvv_path):
        raise SystemExit(f"rvv profile not found: {rvv_path}")
    out = rvv.replace("rvv_x60", "ime_x60")
    out_path = os.path.join(REPO, out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    measured = _load_measured()
    rows = list(csv.DictReader(open(rvv_path)))
    fieldnames = rows[0].keys()
    n_ime = 0
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            op = r.get("op", "")
            if op.startswith("conv2d"):
                d = _parse_shape(r.get("shape", ""))
                try:
                    key = (d["IC"], d["IH"], d["IW"], d["OC"], d["KH"], d["KW"])
                except KeyError:
                    key = None
                sp = measured.get(key) if key else None
                if sp is not None and sp > 1.0:
                    r = dict(r)
                    # scale EVERY cost column by the measured speedup — the loader
                    # reads `mean_time` (load_profiled_times L148), not `cycles`,
                    # so scaling only cycles left the IME cell at the RVV cost.
                    r["cycles"] = str(int(round(float(r["cycles"]) / sp)))
                    for col in ("mean_time", "mean_time_ns"):
                        if r.get(col):
                            r[col] = f"{float(r[col]) / sp:.6f}"
                    r["implementation"] = "curated[ime]/ime_vmadot_4x4x8"
                    r["module_name"] = r["module_name"].replace("rvv_x60", "ime_x60")
                    # PROVENANCE. This cell is a measured RVV cost divided by a measured
                    # speedup -- a prediction, filed in the tree the loader reads as board
                    # measurement. It must not keep the rvv row's `source=k1`, or a
                    # derived cost is indistinguishable from a per-dispatch measurement
                    # (and `--require-source k1` would wave it through).
                    if "source" in r:
                        r["source"] = f"ime_derived({r.get('source') or 'unknown'}/x{sp:g})"
                    n_ime += 1
            w.writerow(r)
    prov = {
        "schema": "ime_derived_profile/v1",
        "kind": "DERIVED, not measured per dispatch",
        "out": out,
        "derived_from": rvv,
        "derived_from_sha256": hashlib.sha256(open(rvv_path, "rb").read()).hexdigest(),
        "speedup_table": os.path.relpath(MEASURED, REPO),
        "speedup_table_sha256": hashlib.sha256(open(MEASURED, "rb").read()).hexdigest(),
        "n_rows": len(rows),
        "n_rows_derived": n_ime,
        "note": ("conv winners carry rvv_cost/measured_speedup with source=ime_derived(...); "
                 "every other row is the rvv measurement copied verbatim so the solver keeps RVV"),
    }
    json.dump(prov, open(os.path.join(os.path.dirname(out_path), "PROVENANCE.json"), "w"),
              indent=1)
    print(f"{args.net}: wrote {out_path}")
    print(f"  {n_ime}/{len(rows)} dispatches given a MEASURED IME cell "
          f"(conv winners); the rest carry the rvv cost so the solver keeps RVV.")
    print(f"  provenance: {n_ime} DERIVED rows tagged source=ime_derived(...); "
          f"sidecar PROVENANCE.json written beside the CSV.")


if __name__ == "__main__":
    main()
