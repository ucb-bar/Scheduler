#!/usr/bin/env python3
"""Generalized ROS per-node-pinning baseline (periodic releases), for arbitrary specs.

Same model as ros_pinning_periodic.py, but reads the net->hart pinning and per-net periods
from the SPEC (and an optional --perception-hart list to CO-LOCATE the vision pipeline on a
single ROS perception executor). Each net is one ROS node; a periodic timer releases each
instance at k*period; the node runs its whole dispatch graph SEQUENTIALLY on its pinned
hart(s) (no per-op cross-hart sharding). Reuses the REAL measured per-dispatch durations
from an XPU-RT schedule of the same workload, so ROS pays the SAME per-op costs as XPU-RT --
only the serialization policy differs.
"""
import argparse, json, os, re, collections, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
from job_names import split_job_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="XPU-RT schedule JSON (source of real per-dispatch durations)")
    ap.add_argument("--spec", required=True, help="networks spec JSON (periods + net list)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--perception-nets", default="yolov8_nano_64x96,ffn_block,attn_block",
                    help="comma-list of nets co-located on ONE perception executor hart")
    ap.add_argument("--control-hart-base", type=int, default=0)
    a = ap.parse_args()

    spec = json.load(open(a.spec))["networks"]
    known = set(spec)
    period = {n: float(v.get("period", 0) or 0) for n, v in spec.items()}
    perception = [n for n in a.perception_nets.split(",") if n in spec]

    # Assign one hart per non-perception net; all perception nets share ONE executor hart.
    net_hart = {}
    h = a.control_hart_base
    for n in spec:
        if n in perception:
            continue
        net_hart[n] = f"CPU_P#{h}"; h += 1
    perc_hart = f"CPU_P#{h}"
    for n in perception:
        net_hart[n] = perc_hart

    src = json.load(open(a.src if os.path.isabs(a.src) else os.path.join(REPO, a.src)))
    items = list(src["dispatches"].values())
    # Sort by (net, numeric instance, dispatch ordinal) so instance 10 does NOT string-sort
    # before instance 2 (which would corrupt the serial per-hart release timing).
    def _key(z):
        net, inst = split_job_name(z.get("job_name", ""), known)
        return (net, int(inst), z.get("ordinal", z.get("id", 0)))
    by_job = collections.OrderedDict()
    for x in sorted(items, key=_key):
        by_job.setdefault(x["job_name"], []).append(x)

    hart_free = collections.defaultdict(float)
    out = {}
    nid = 0
    for job, ds in by_job.items():
        net, inst = split_job_name(job, known)
        hart = net_hart[net]; per = period[net]
        t = max(inst * per, hart_free[hart])
        for x in ds:
            nid += 1
            out[str(nid)] = {"id": nid, "ordinal": x.get("ordinal", 1), "total": x.get("total", 1),
                             "dependencies": [], "hardware_target": hart,
                             "start_time": round(t, 6), "duration": x["duration"],
                             "job_name": job, "module_name": x.get("module_name", job),
                             "release_policy": "periodic", "time_dep_mode": "hard"}
            t += x["duration"]
        hart_free[hart] = t
    mk = max(v["start_time"] + v["duration"] for v in out.values())
    res = {"metadata": {"makespan": mk, "policy": "ros_pinning_generic",
                        "perception_executor_hart": perc_hart,
                        "perception_nets": perception, "net_hart": net_hart},
           "dispatches": out}
    json.dump(res, open(a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out), "w"), indent=1)
    print("wrote %s: %d dispatches, makespan %.2f ms | perception hart %s <- %s"
          % (a.out, len(out), mk, perc_hart, perception))


if __name__ == "__main__":
    main()
