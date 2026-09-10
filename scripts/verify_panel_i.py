#!/usr/bin/env python3
"""Verify panel I of warehouse_showdown_story: the XPU-RT vs ROS onboard Gantt.

WHAT PANEL I PLOTS, AND THE DEFECT. The two schedules it draws are generated over
DIFFERENT RELEASE HORIZONS. XPU-RT's last mlp_control release is at 30 ms (4 instances at a
10 ms period); the ROS schedule's is at 110 ms (12 instances). So "XPU-RT done at 40 ms,
ROS still backlogged at 112 ms" contrasts a 40 ms schedule with a 120 ms one: the 112 ms is
where ROS's releases END, not evidence that it is 2.8x slower. Normalized, the two keep
1.42 and 1.29 cores busy -- 11% apart.

This script truncates the ROS schedule to XPU-RT's horizon and rescores. Truncation is exact
for this model: each net's instances run sequentially on its pinned hart in release order, so
dropping instances k >= N cannot change the placement of k < N. It preserves the plotted
policy exactly -- yolo on its 4-hart shard, mlp on CPU_E#0, fused on CPU_E#1, 6 cores.

  XPU-RT (CP-SAT, 8 cores)        40.38 ms   57.4 core-ms   0 misses
  ROS truncated (same horizon)    46.35 ms   56.3 core-ms   2 misses  (2/2 yolo instances)
  ROS as plotted (3x horizon)    112.35 ms  144.5 core-ms   5 misses  (5/5 yolo instances)

So the honest makespan ratio is 1.15x, not 2.8x. The claim that does NOT depend on the
horizon -- and is therefore the one to lead with -- is the deadline result: ROS misses
100% of its perception instances at both horizons, XPU-RT misses none.

WHAT THIS IS NOT. No ROS software runs here, and none ran in the figure. "ROS" throughout is
scripts/ros_pinning_generic.py, a SERIALIZATION POLICY MODEL costed from XPU-RT's own
measured per-dispatch durations. It pays the same per-op costs, so the comparison isolates
placement policy -- but it is a model of ROS, not a measurement of it. See
docs/measurements_and_ablations.md section 4.

NOTE ON PARSING. Instance indices MUST come from split_job_name(job, known) with the known
net set. The one-argument form strips all trailing digits, and `yolov8_nano_64x96` ends in
digits: `yolov8_nano_64x960` parses as ('yolov8_nano_64x', 960) instead of
('yolov8_nano_64x96', 0), which silently drops every yolo instance from a filter.
"""
import argparse, collections, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
from job_names import split_job_name          # noqa: E402
from schedule_eval import summary             # noqa: E402


def instances(path, known):
    d = json.load(open(path))["dispatches"]
    by = collections.defaultdict(set)
    for v in d.values():
        n, i = split_job_name(v["job_name"], known)
        by[n].add(i)
    return {n: sorted(s) for n, s in by.items()}


def stats(path, spec):
    d = json.load(open(path))["dispatches"]
    mk = max(v["start_time"] + v["duration"] for v in d.values())
    wk = sum(v["duration"] for v in d.values())
    s = summary(path, spec)
    return mk, wk, len(d), s["instance_misses"], s["misses_by_network"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="data/toplevel/_flight_deployed_2frame.json")
    ap.add_argument("--xpu", default="schedules/scheduled__flight_deployed_2frame_cpsat_profiled.json")
    ap.add_argument("--ros", default="schedules/scheduled_ros_partition_deployed.json")
    ap.add_argument("--out", default="/tmp/ros_truncated.json", help="the truncated ROS schedule")
    a = ap.parse_args()
    for p in (a.spec, a.xpu, a.ros):
        if not os.path.isabs(p) and not os.path.exists(p):
            os.chdir(REPO)
    known = set(json.load(open(a.spec))["networks"])

    ix, ir = instances(a.xpu, known), instances(a.ros, known)
    print("XPU-RT instances:", {n: len(v) for n, v in sorted(ix.items())})
    print("ROS    instances:", {n: len(v) for n, v in sorted(ir.items())})
    if {n: len(v) for n, v in ix.items()} == {n: len(v) for n, v in ir.items()}:
        print("\nHorizons already match -- nothing to correct.")
        return 0
    print("\nHORIZON MISMATCH: the two schedules do not cover the same releases.")

    want = {n: len(v) for n, v in ix.items()}
    ros = json.load(open(a.ros))["dispatches"]
    keep = {}
    for k, v in ros.items():
        n, i = split_job_name(v["job_name"], known)
        if n in want and i < want[n]:
            keep[k] = v
    json.dump({"metadata": {"policy": "ros_partition, truncated to the XPU-RT release horizon"},
               "dispatches": keep}, open(a.out, "w"), indent=1)

    print(f"\n{'schedule':<34} {'makespan':>11} {'work':>12} {'disp':>6}  misses")
    rows = [("XPU-RT (CP-SAT)", a.xpu), ("ROS truncated (same horizon)", a.out),
            ("ROS as plotted", a.ros)]
    out = {}
    for tag, p in rows:
        mk, wk, n, m, by = stats(p, a.spec)
        out[tag] = mk
        print(f"{tag:<34} {mk:>9.2f}ms {wk:>9.1f}cms {n:>6}  {m} {by}")
    r = out["ROS truncated (same horizon)"] / out["XPU-RT (CP-SAT)"]
    print(f"\nHonest makespan ratio, same horizon and same policy: {r:.2f}x")
    print("Horizon-independent claim: ROS misses 100% of perception instances at BOTH "
          "horizons (2/2 truncated, 5/5 as plotted); XPU-RT misses none.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
