"""Fast shared algorithm (milestone M2), used by backends/openmp.py and backends/mpi.py
(and the merge step of backends/opencl.py).

The idea (docs/ARCHITECTURE.md section 6): with failures sorted by (user, ts), for failure i

    j_i     = lower_bound(ts, ts_i - W + 1)   # first failure inside the window (t - W, t]
    count_i = i - j_i + 1
    trigger = count_i >= N

Each count depends only on the sorted array, not on the previous result, so the
index range can be split in any way (threads, processes, GPU work-items) and the
result stays the same.

Pipeline (detect() below runs it in one thread; the backends reuse the pieces):

    sort_failures    filter FAIL events, stable sort by (user, ts), user segments
    window_counts    j and count for every failure         <- the parallel part (prange)
    find_episodes    triggers -> alarms, per user           (sequential inside a user)
    alarms_frame     alarm arrays -> DataFrame
    build_summary    per-account counts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

from login_correlation.common import (
    DetectionResult,
    EventArrays,
    PhaseTimer,
    RuleParams,
    normalize_alarms,
    normalize_summary,
)

# ----------------------------------------------------------------------------- 1. sort


def sort_failures(events: EventArrays) -> tuple[np.ndarray, np.ndarray]:
    """Keep the failures, sort them by (user, ts) and find each user's segment.

    Returns
        ts_sorted   int64[F]    failure timestamps; user u owns ts_sorted[off[u]:off[u+1]]
        seg_offsets int64[U+1]  segment borders, one entry per account in events.users (+1)

    np.lexsort is stable, so failures with the same (user, ts) keep their file order (seq).
    """
    mask = events.is_fail.astype(bool)
    ts_f = events.ts[mask]
    uid_f = events.user_id[mask]
    order = np.lexsort((ts_f, uid_f))  # last key = primary key -> sort by user, then ts
    per_user = np.bincount(uid_f, minlength=len(events.users))
    seg_offsets = np.zeros(len(events.users) + 1, dtype=np.int64)
    np.cumsum(per_user, out=seg_offsets[1:])
    return ts_f[order].astype(np.int64), seg_offsets


# ----------------------------------------------------------------------------- 2. count


@njit(cache=True)
def lower_bound(a: np.ndarray, lo: int, hi: int, value: int) -> int:
    """First index k in [lo, hi) with a[k] >= value (hi if there is none)."""
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(parallel=True, cache=True)
def window_counts(ts_sorted, seg_offsets, window_s, order):
    """Return (j, count) for every failure.

    ts_sorted   int64[F]   failure timestamps sorted by (user, ts), stable
    seg_offsets int64[U+1] user u owns ts_sorted[seg_offsets[u]:seg_offsets[u+1]]
    order       int64[U]   order in which users are handed to threads (load balancing)

    Each failure i writes only j[i] and count[i], so threads never write the same place.
    """
    F = ts_sorted.shape[0]
    j_out = np.empty(F, dtype=np.int64)
    count_out = np.empty(F, dtype=np.int64)
    for k in prange(order.shape[0]):  # <- the OpenMP parallel loop
        u = order[k]
        lo = seg_offsets[u]
        for i in range(lo, seg_offsets[u + 1]):
            # first failure of this user with ts > ts_i - W, searched only in [lo, i]
            j = lower_bound(ts_sorted, lo, i + 1, ts_sorted[i] - window_s + 1)
            j_out[i] = j
            count_out[i] = i - j + 1
    return j_out, count_out


# ----------------------------------------------------------------------------- 3. episodes


@njit(cache=True)
def find_episodes(seg_offsets, j, count, threshold):
    """Merge triggers into alarms (docs/ARCHITECTURE.md section 2.3).

    For each user, walk the failures in order. A trigger (count >= N) whose window start
    j_i is <= the last index of the open alarm extends that alarm; otherwise it closes the
    open alarm and starts a new one.

    Returns arrays, one entry per alarm:
        ep_user   account id
        first_j   index of the first failure of the first triggering window -> start_ts
        first_i   index of the first trigger                                 -> trigger_ts
        last_i    index of the last trigger                                  -> end_ts
    and max_count[u] = largest count of user u (for the summary).
    """
    U = seg_offsets.shape[0] - 1
    F = j.shape[0]
    ep_user = np.empty(F, dtype=np.int64)  # there cannot be more alarms than failures
    first_j = np.empty(F, dtype=np.int64)
    first_i = np.empty(F, dtype=np.int64)
    last_i = np.empty(F, dtype=np.int64)
    max_count = np.zeros(U, dtype=np.int64)
    n = 0
    for u in range(U):
        is_open = False
        for i in range(seg_offsets[u], seg_offsets[u + 1]):
            if count[i] > max_count[u]:
                max_count[u] = count[i]
            if count[i] < threshold:
                continue
            if is_open and j[i] <= last_i[n - 1]:
                last_i[n - 1] = i  # window shares a failure with the open alarm
            else:
                ep_user[n] = u
                first_j[n] = j[i]
                first_i[n] = i
                last_i[n] = i
                n += 1
                is_open = True
    return ep_user[:n], first_j[:n], first_i[:n], last_i[:n], max_count


def alarms_frame(ts_sorted, ep_user, first_j, first_i, last_i, users: list[str]) -> pd.DataFrame:
    """Alarm arrays -> DataFrame with common.ALARM_COLUMNS, in the canonical order."""
    names = np.asarray(users, dtype=object)[ep_user]
    start_ts = ts_sorted[first_j].astype(np.int64)
    df = pd.DataFrame(
        {
            "alarm_id": pd.Series(names, dtype=object) + ":" + start_ts.astype(str).astype(object),
            "user": pd.Series(names, dtype=object),
            "start_ts": start_ts,
            "end_ts": ts_sorted[last_i].astype(np.int64),
            "trigger_ts": ts_sorted[first_i].astype(np.int64),
            "n_failures": (last_i - first_j + 1).astype(np.int64),
        }
    )
    df["alarm_id"] = df["alarm_id"].astype(object)
    return normalize_alarms(df)


# ----------------------------------------------------------------------------- 4. summary


@njit(cache=True)
def _user_stats(user_id, ts, n_users):
    """One pass over all events: n_events, first_ts, last_ts per user."""
    n_events = np.zeros(n_users, dtype=np.int64)
    first = np.zeros(n_users, dtype=np.int64)
    last = np.zeros(n_users, dtype=np.int64)
    for k in range(user_id.shape[0]):
        u = user_id[k]
        t = ts[k]
        if n_events[u] == 0 or t < first[u]:
            first[u] = t
        if n_events[u] == 0 or t > last[u]:
            last[u] = t
        n_events[u] += 1
    return n_events, first, last


def build_summary(events: EventArrays, max_count: np.ndarray, ep_user: np.ndarray) -> pd.DataFrame:
    """Per-account summary (docs/ARCHITECTURE.md section 2.4), one row per entry in events.users."""
    U = len(events.users)
    n_events, first, last = _user_stats(events.user_id, events.ts, U)
    n_fail = np.bincount(events.user_id, weights=events.is_fail, minlength=U).astype(np.int64)
    return summary_frame(events.users, n_events, n_fail, first, last, max_count, ep_user)


def summary_frame(users, n_events, n_fail, first, last, max_count, ep_user) -> pd.DataFrame:
    """Per-user arrays -> summary DataFrame. Shared by build_summary (one thread) and
    backends/openmp.py (statistics computed in parallel)."""
    U = len(users)
    no_events = n_events == 0  # empty accounts: first_ts / last_ts are empty (NA)
    df = pd.DataFrame(
        {
            "user": pd.Series(users, dtype=object),
            "n_events": n_events,
            "n_fail": n_fail,
            "n_success": n_events - n_fail,
            "first_ts": pd.arrays.IntegerArray(first, no_events.copy()),
            "last_ts": pd.arrays.IntegerArray(last, no_events.copy()),
            "max_window_count": max_count.astype(np.int64),
            "n_alarms": np.bincount(ep_user, minlength=U).astype(np.int64),
        }
    )
    return normalize_summary(df)


# ----------------------------------------------------------------------------- whole pipeline


def detect(events: EventArrays, params: RuleParams, order: np.ndarray | None = None) -> DetectionResult:
    """Run the whole pipeline. With Numba's default thread count the prange in window_counts
    is already parallel; backends/openmp.py sets the thread count and the user order."""
    timer = PhaseTimer()
    with timer.phase("total"):
        with timer.phase("prepare"):
            ts_sorted, seg_offsets = sort_failures(events)
            if order is None:
                order = np.arange(len(events.users), dtype=np.int64)
        with timer.phase("compute"):
            j, count = window_counts(ts_sorted, seg_offsets, params.window_s, order)
        with timer.phase("merge"):
            ep_user, first_j, first_i, last_i, max_count = find_episodes(
                seg_offsets, j, count, params.threshold
            )
            alarms = alarms_frame(ts_sorted, ep_user, first_j, first_i, last_i, events.users)
            summary = build_summary(events, max_count, ep_user)
    return DetectionResult(alarms, summary, timer.as_dict())


def warm_up() -> float:
    """Compile all Numba functions on a tiny input and return the compile time in seconds.
    Call it before timing, so compilation is not counted as computation."""
    import time

    start = time.perf_counter()
    tiny = EventArrays(
        ts=np.array([0, 1, 2], dtype=np.int64),
        user_id=np.array([0, 0, 1], dtype=np.int32),
        ip_id=np.zeros(3, dtype=np.int32),
        is_fail=np.array([1, 1, 0], dtype=np.uint8),
        users=["a", "b"],
    )
    detect(tiny, RuleParams(60, 2))
    return time.perf_counter() - start
