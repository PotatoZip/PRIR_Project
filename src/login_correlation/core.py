"""Fast shared algorithm, used by backends/openmp.py and backends/mpi.py (and the merge step of backends/opencl.py)

The idea (docs/ARCHITECTURE.md section 6): with failures sorted by (user, ts), for failure i

    j_i     = lower_bound(ts, ts_i - W + 1)   # first failure inside the window
    count_i = i - j_i + 1
    trigger = count_i >= N

Each count depends only on the sorted array, not on the previous result, so the
index range can be split in any way (threads, processes, GPU work-items) and the
result stays the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

from login_correlation.common import EventArrays


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
    """
    F = ts_sorted.shape[0]
    j_out = np.empty(F, dtype=np.int64)
    count_out = np.empty(F, dtype=np.int64)
    for k in prange(order.shape[0]):  # <- the OpenMP parallel loop
        u = order[k]
        for i in range(seg_offsets[u], seg_offsets[u + 1]):
            # TODO(M2): j = lower_bound(ts_sorted, seg_offsets[u], i + 1, ts_sorted[i] - window_s + 1)
            j_out[i] = i
            count_out[i] = 1
    return j_out, count_out


def merge_episodes(
    ts_sorted, seg_offsets, j, count, threshold: int, users: list[str]
) -> pd.DataFrame:
    """Triggers -> alarms (rule in docs/ARCHITECTURE.md section 2.3). TODO(M2).

    For each user, walk the triggers in order: a trigger whose window start j_i is
    <= last_i of the open alarm extends it, otherwise it opens a new alarm.
    Return common.normalize_alarms(DataFrame with common.ALARM_COLUMNS).
    """
    raise NotImplementedError("core.merge_episodes - milestone M2")


def build_summary(
    events: EventArrays, max_window_count: np.ndarray, alarms: pd.DataFrame
) -> pd.DataFrame:
    """Per-account summary, one row per entry in events.users

    Hint: np.bincount(user_id, minlength=len(users)) -> n_events; with weights=is_fail -> n_fail;
    np.minimum.at / np.maximum.at -> first_ts / last_ts. Return common.normalize_summary(...).
    """
    raise NotImplementedError("core.build_summary - milestone M2")
