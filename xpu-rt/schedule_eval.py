"""Instance-level evaluation of a solved schedule: misses, lateness, response.

WHY INSTANCE-LEVEL. The schedulers report `op_deadline_miss_count`, a per-DISPATCH
number. What a real-time claim is about is whether a network INSTANCE finished inside
its window -- an instance misses when its LATEST dispatch ends past
`inst*period + window`. Those two numbers differ, and the figures, the loop's
board-feedback arm and the ablation all have to use the same one or they are not
describing the same experiment.

Extracted from run_codesign_loop.py so the loop, the ablation and anything else score a
schedule identically rather than each carrying a copy that drifts.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

from job_names import split_job_name


def _last_end_per_instance(sched_path: str, spec):
    """`({(net, inst): end_ms}, periods, windows)` or `None` if unreadable."""
    try:
        sch = json.load(open(sched_path))["dispatches"]
        nets = spec["networks"] if isinstance(spec, dict) else json.load(
            open(spec))["networks"]
    except Exception:
        return None
    known = set(nets)
    per = {n: float(v.get("period", 0) or 0) for n, v in nets.items()}
    win = {n: float(v.get("window_duration", 0) or 0) for n, v in nets.items()}
    last: Dict[Tuple[str, int], float] = {}
    for d in sch.values():
        net, inst = split_job_name(d["job_name"], known)
        if not (net in per and per[net]):
            continue
        e = float(d["start_time"]) + float(d["duration"])
        last[(net, inst)] = max(last.get((net, inst), 0.0), e)
    return last, per, win


def instance_misses(sched_path: str, spec) -> Tuple[Optional[int], dict]:
    """`(count, {net: count})` of net-instances that ended past their deadline."""
    got = _last_end_per_instance(sched_path, spec)
    if got is None:
        return None, {"error": "unreadable schedule or spec"}
    last, per, win = got
    miss, by = 0, {}
    for (net, inst), e in last.items():
        if win[net] > 0 and e > inst * per[net] + win[net] + 1e-6:
            miss += 1
            by[net] = by.get(net, 0) + 1
    return miss, by


def total_lateness(sched_path: str, spec) -> Optional[float]:
    """Sum over instances of `max(0, end - deadline)`, ms. Zero iff every instance met.

    Credits a lever that pulls an instance in ahead of its deadline even when the
    makespan is unchanged -- which is exactly how IME helps on some workloads.
    """
    got = _last_end_per_instance(sched_path, spec)
    if got is None:
        return None
    last, per, win = got
    total = 0.0
    for (net, inst), e in last.items():
        if win[net] > 0:
            total += max(0.0, e - (inst * per[net] + win[net]))
    return total


def worst_lateness(sched_path: str, spec) -> Optional[float]:
    got = _last_end_per_instance(sched_path, spec)
    if got is None:
        return None
    last, per, win = got
    worst = 0.0
    for (net, inst), e in last.items():
        if win[net] > 0:
            worst = max(worst, e - (inst * per[net] + win[net]))
    return worst


def makespan_ms(sched_path: str) -> Optional[float]:
    """Makespan from the schedule's `_metrics.json` sidecar."""
    try:
        m = json.load(open(sched_path.replace(".json", "_metrics.json")))
        return float(m.get("makespan_ms", m.get("makespan", 0.0)) or 0.0)
    except Exception:
        return None


def summary(sched_path: str, spec) -> dict:
    """Everything the ablation reports for one schedule, on one set of costs."""
    miss, by = instance_misses(sched_path, spec)
    return {
        "instance_misses": miss,
        "misses_by_network": by,
        "total_lateness_ms": total_lateness(sched_path, spec),
        "worst_lateness_ms": worst_lateness(sched_path, spec),
        "makespan_ms": makespan_ms(sched_path),
    }
