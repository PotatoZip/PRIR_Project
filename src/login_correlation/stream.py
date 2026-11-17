"""Stream replay and time-to-alarm (milestone M6).

Run:  python -m login_correlation replay --data data\\demo --speedup 10000

Batch mode (`detect`) reads the whole file and reports alarms at the end. A real security
system sees login events one by one and must raise an alarm as soon as an attack is visible.
This module simulates that:

1. The events are REPLAYED in time order (ts, then file order), as if they arrived live.
   `speedup` = how many seconds of log time pass in one second of real time
   (1 = real time, 10000 = a week of logs in about a minute, inf = as fast as possible).
2. Each event is handled the moment it "arrives". For every account the detector keeps a
   queue (deque) of its failures from the last W seconds; older ones are dropped from the
   front. When the queue reaches N failures, an alarm is RAISED immediately.
3. The rule is exactly the batch rule (docs/ARCHITECTURE.md section 2): the window is
   (t - W, t], equal timestamps keep file order, and triggers that share a failure with
   the open alarm extend it instead of raising a new one. So the final list of alarms
   must be identical to the batch result - this is tested.

What is measured (per alarm, then p50 / p95 / max):

    processing latency  real time from the arrival of the event that completes the attack
                        (the N-th failure) to the alarm. Shows how fast the code reacts.
    time to alarm       real time from the arrival of the first failure in the window to the
                        alarm = (trigger_ts - start_ts) / speedup + processing latency.
                        The "czas do uzyskania alarmu w odtwarzanym strumieniu" of the task.
    detection delay     the same in log time: trigger_ts - start_ts (at most W - 1 seconds,
                        because the N-th failure has to arrive first)
    lag                 how late events are handled compared with their scheduled arrival.
                        A small max lag means the detector keeps up with the stream.

The loop is plain Python on purpose: a stream is handled one event at a time, in order,
so there is nothing to split between threads, and the per-event work is tiny.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import pandas as pd

from login_correlation.common import EventArrays, RuleParams, make_alarm_id, normalize_alarms


def _wait_until(t: float) -> None:
    """Wait until perf_counter() >= t: sleep for the long part, spin for the last ~1.5 ms
    (sleep alone can wake up a few milliseconds late, which would show up as lag)."""
    while True:
        rest = t - time.perf_counter()
        if rest <= 0:
            return
        if rest > 0.002:
            time.sleep(rest - 0.0015)


def _percentiles(values: list[float], scale: float = 1.0) -> dict:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    a = np.asarray(values) * scale
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}


def replay(events: EventArrays, params: RuleParams, speedup: float = math.inf) -> dict:
    """Replay the events as a live stream. Returns

        alarms     DataFrame, the same columns and order as the batch backends
        per_alarm  DataFrame, one row per alarm with the times measured when it was raised
        stats      dict: events, alarms, wall time, events/s, lag, latency percentiles
    """
    if not speedup > 0:
        raise ValueError("speedup must be > 0 (use inf for 'as fast as possible')")
    W, N = params.window_s, params.threshold
    paced = math.isfinite(speedup)

    order = np.argsort(events.ts, kind="stable")       # time order; ties keep file order
    ts_list = events.ts[order].tolist()
    uid_list = events.user_id[order].tolist()
    fail_list = events.is_fail[order].tolist()
    ts0 = ts_list[0] if ts_list else 0

    window: dict[int, deque] = {}     # user -> deque of (failure index, ts) in the window
    n_fail: dict[int, int] = {}       # user -> failures seen so far (gives the index)
    open_alarm: dict[int, list] = {}  # user -> [start_idx, start_ts, trigger_ts, last_idx, end_ts]
    closed: list[tuple] = []          # (user, start_idx, start_ts, trigger_ts, last_idx, end_ts)
    raised: list[dict] = []
    arrival_of: dict[tuple, float] = {}  # (user, failure index) -> arrival time, while in a window
    max_lag = 0.0

    t_start = time.perf_counter()
    for ts, u, f in zip(ts_list, uid_list, fail_list):
        scheduled = t_start + (ts - ts0) / speedup if paced else None
        if paced:
            _wait_until(scheduled)
        arrival = time.perf_counter()
        if paced:
            max_lag = max(max_lag, arrival - scheduled)
        if not f:
            continue                    # successful logins do not affect the rule

        # ---- the rule, one failure at a time
        i = n_fail.get(u, 0)
        n_fail[u] = i + 1
        q = window.get(u)
        if q is None:
            q = window[u] = deque()
        while q and q[0][1] <= ts - W:  # drop failures at or before ts - W
            old = q.popleft()
            arrival_of.pop((u, old[0]), None)
        q.append((i, ts))
        arrival_of[(u, i)] = scheduled if paced else arrival
        if len(q) < N:
            continue
        j, start_ts = q[0]
        alarm = open_alarm.get(u)
        if alarm is not None and j <= alarm[3]:
            alarm[3], alarm[4] = i, ts  # extends the open alarm (shares a failure with it)
            continue
        if alarm is not None:
            closed.append((u, *alarm))
        open_alarm[u] = [j, start_ts, ts, i, ts]
        emitted = time.perf_counter()  # <- the alarm is raised here
        raised.append({
            "user": events.users[u], "start_ts": start_ts, "trigger_ts": ts,
            "processing_latency_s": emitted - arrival,
            "time_to_alarm_s": emitted - arrival_of[(u, j)],
            "detection_delay_s": ts - start_ts,
        })
    wall = time.perf_counter() - t_start
    closed.extend((u, *a) for u, a in open_alarm.items())

    alarms = normalize_alarms(pd.DataFrame([{
        "alarm_id": make_alarm_id(events.users[u], start_ts),
        "user": events.users[u],
        "start_ts": start_ts,
        "end_ts": end_ts,
        "trigger_ts": trigger_ts,
        "n_failures": last_idx - start_idx + 1,
    } for u, start_idx, start_ts, trigger_ts, last_idx, end_ts in closed]))
    per_alarm = pd.DataFrame(raised, columns=["user", "start_ts", "trigger_ts", "processing_latency_s",
                                              "time_to_alarm_s", "detection_delay_s"])
    stats = {
        "events": len(ts_list),
        "alarms": len(alarms),
        "speedup": speedup if paced else "inf",
        "log_span_s": (ts_list[-1] - ts0) if ts_list else 0,
        "wall_s": wall,
        "events_per_s": len(ts_list) / wall if wall > 0 else None,
        "max_lag_ms": max_lag * 1000 if paced else None,
        "processing_latency_ms": _percentiles(per_alarm["processing_latency_s"].tolist(), 1000),
        "time_to_alarm_s": _percentiles(per_alarm["time_to_alarm_s"].tolist()),
        "detection_delay_s": _percentiles(per_alarm["detection_delay_s"].tolist()),
    }
    return {"alarms": alarms, "per_alarm": per_alarm, "stats": stats}


def format_report(stats: dict, identical: bool | None = None) -> str:
    """Human-readable summary printed by `python -m login_correlation replay`."""
    def fmt(d, unit, nd=3):
        if d["p50"] is None:
            return "-"
        return f"p50 {d['p50']:.{nd}f} {unit}, p95 {d['p95']:.{nd}f} {unit}, max {d['max']:.{nd}f} {unit}"

    sp = stats["speedup"]
    lines = [
        f"replayed          {stats['events']:,} events ({stats['log_span_s']:,} s of log time) "
        f"in {stats['wall_s']:.2f} s, speed-up {sp if sp == 'inf' else f'{sp:g}x'}",
        f"throughput        {stats['events_per_s']:,.0f} events/s" if stats["events_per_s"] else "throughput        -",
        f"alarms            {stats['alarms']}",
        f"processing        {fmt(stats['processing_latency_ms'], 'ms')}   "
        "(N-th failure arrives -> alarm)",
        f"time to alarm     {fmt({k: (v * 1000 if v is not None else None) for k, v in stats['time_to_alarm_s'].items()}, 'ms')}   "
        "(first failure in the window arrives -> alarm, real time)",
        f"detection delay   {fmt(stats['detection_delay_s'], 's', 0)}   (the same in log time)",
    ]
    if stats["max_lag_ms"] is not None:
        lines.append(f"max lag           {stats['max_lag_ms']:.2f} ms   (how late the slowest event "
                     "was handled - small = the detector keeps up)")
    if identical is not None:
        lines.append(f"same alarms as batch detection: {'YES' if identical else 'NO'}")
    return "\n".join(lines)
