"""OpenMP backend: Numba `prange` loops running on the OpenMP threading layer (milestone M3).

Design: docs/ARCHITECTURE.md section 7.1. The task names two ways to split the work and
both are implemented, chosen with `split`:

    split="users"  accounts between threads (variant A)
    split="time"   groups of time-ordered events between threads, with a halo (variant B)

Pipeline and what runs in parallel (P = on the OpenMP threads):

    prepare  P  group_by_user           radix partitioning: events into blocks of users, then
                                        each block -> one thread: statistics + sorted failures
             P  balanced_order          (users) biggest users spread over the threads
                time_order + time_cuts  (time)  failures in time order, cut into chunks
    compute  P  core.window_counts      (users) prange over users
             P  time_split_counts       (time)  prange over time chunks, each reads a halo
    merge       core.find_episodes      one pass over the failures (small)
                core.alarms_frame / summary_frame   build the output tables (pandas)

Every parallel loop writes only to its own part of an array (its user segment, its thread's
row, or the failures it owns), so no locks or atomics are needed.

Design note - how group_by_user got here (10 M events, 100 000 users, docs/ROADMAP.md M3):
  1. per-thread partial arrays (T x U) combined at the end: 38 MB of scattered writes with
     12 threads, far bigger than the 12 MB CPU cache -> SLOWER with more threads.
  2. full counting sort of all events by user: correct, but writing 90 MB to random places
     is limited by memory speed -> no gain after 2 threads.
  3. "owner computes" (each thread scans ALL events, handles only its users): the test
     "is this my user?" is random, the CPU mispredicts it -> 4x slower per thread.
  4. radix partitioning (this): first sort the events into a few blocks of users (cheap,
     cache-friendly), then each thread owns whole blocks and reads only their events.

Numba compiles the functions the first time they run. That time is measured once and
reported separately as "compile" (it is not part of "total").
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")  # must be set before Numba starts its threads

import numba  # noqa: E402
import numpy as np  # noqa: E402
from numba import njit, prange  # noqa: E402

from login_correlation import core  # noqa: E402
from login_correlation.common import DetectionResult, EventArrays, PhaseTimer, RuleParams  # noqa: E402

# ----------------------------------------------------------------------------- prepare


BLOCK_USERS = 8192  # users per block: 4 arrays x 8192 x 8 B = 256 KB, fits a core's L2 cache


@njit(parallel=True, cache=True)
def group_by_user(user_id, ts, is_fail, n_users, n_threads):
    """Group the failures by user and compute the per-user statistics, in parallel.

    Pattern: radix partitioning ("partition first, then owner computes").
    The users are cut into B blocks of at most BLOCK_USERS consecutive ids.

    1. histogram  (prange over event chunks): each thread counts how many of its events fall
                  into each block. The table is only T x B (tiny), so it stays in cache.
    2. prefix sum (sequential, B x T steps): where each thread writes each block.
    3. scatter    (prange over event chunks): each thread copies its events into the blocks.
                  It writes to only B places that move forward in order (not to random places).
                  Within a block the events keep their original order.
    4. per block  (prange over blocks): one thread OWNS the block. It reads only the block's
                  events, in order, and fills the statistics of the block's users
                  (n_events, n_fail, first_ts, last_ts), then copies their failure timestamps
                  to the layout core.find_episodes expects (seg_offsets) and sorts a user's
                  segment only if it is not already in time order (sorting in place).

    No thread ever writes where another thread writes, so no locks are needed, and there is
    no final combining step. Returns ts_sorted, seg_offsets (as core.sort_failures) and
    n_events, n_fail, first, last per user.
    """
    n = user_id.shape[0]
    T = n_threads
    U = n_users
    B = max(1, min(max(U, 1), max(4 * T, (U + BLOCK_USERS - 1) // BLOCK_USERS)))
    bs = max(1, (U + B - 1) // B)  # users per block
    chunk = (n + T - 1) // T

    # 1. histogram of events and failures per (thread, block)
    hist = np.zeros((T, B), dtype=np.int64)
    hist_f = np.zeros((T, B), dtype=np.int64)
    for t in prange(T):
        for k in range(t * chunk, min(n, (t + 1) * chunk)):
            b = user_id[k] // bs
            hist[t, b] += 1
            hist_f[t, b] += is_fail[k]

    # 2. prefix sums: block offsets and each thread's write position inside each block
    blk_off = np.zeros(B + 1, dtype=np.int64)
    blk_fail_off = np.zeros(B + 1, dtype=np.int64)
    write_at = np.zeros((T, B), dtype=np.int64)
    pos = 0
    fpos = 0
    for b in range(B):
        blk_off[b] = pos
        blk_fail_off[b] = fpos
        for t in range(T):
            write_at[t, b] = pos
            pos += hist[t, b]
            fpos += hist_f[t, b]
    blk_off[B] = pos
    blk_fail_off[B] = fpos

    # 3. scatter the events into their blocks
    p_uid = np.empty(n, dtype=np.int32)
    p_ts = np.empty(n, dtype=np.int64)
    p_fail = np.empty(n, dtype=np.uint8)
    for t in prange(T):
        for k in range(t * chunk, min(n, (t + 1) * chunk)):
            b = user_id[k] // bs
            w = write_at[t, b]
            p_uid[w] = user_id[k]
            p_ts[w] = ts[k]
            p_fail[w] = is_fail[k]
            write_at[t, b] = w + 1

    # 4. each block is handled by one thread
    n_events = np.zeros(U, dtype=np.int64)
    n_fail = np.zeros(U, dtype=np.int64)
    first = np.zeros(U, dtype=np.int64)
    last = np.zeros(U, dtype=np.int64)
    seg_offsets = np.zeros(U + 1, dtype=np.int64)
    seg_offsets[U] = fpos
    ts_sorted = np.empty(fpos, dtype=np.int64)
    for b in prange(B):
        u0 = b * bs
        u1 = min(U, (b + 1) * bs)
        if u0 >= u1:
            continue
        for k in range(blk_off[b], blk_off[b + 1]):
            u = p_uid[k]
            if n_events[u] == 0 or p_ts[k] < first[u]:
                first[u] = p_ts[k]
            if n_events[u] == 0 or p_ts[k] > last[u]:
                last[u] = p_ts[k]
            n_events[u] += 1
            n_fail[u] += p_fail[k]
        f = blk_fail_off[b]
        for u in range(u0, u1):
            seg_offsets[u] = f
            f += n_fail[u]
        fill = seg_offsets[u0:u1].copy()
        for k in range(blk_off[b], blk_off[b + 1]):
            if p_fail[k]:
                u = p_uid[k]
                ts_sorted[fill[u - u0]] = p_ts[k]
                fill[u - u0] += 1
        # Logs are usually in time order already: check first, sort in place only if needed
        for u in range(u0, u1):
            a = seg_offsets[u]
            e = a + n_fail[u]
            for k in range(a + 1, e):
                if ts_sorted[k] < ts_sorted[k - 1]:
                    ts_sorted[a:e].sort()
                    break
    return ts_sorted, seg_offsets, n_events, n_fail, first, last


def balanced_order(seg_offsets: np.ndarray, n_threads: int) -> np.ndarray:
    """User order for the prange in core.window_counts.

    Numba gives each thread one contiguous block of iterations (like OpenMP
    schedule(static)). Users sorted by size and dealt round-robin (thread 0 gets the 1st,
    (T+1)-th, ... biggest user) give every block about the same amount of work.
    """
    sizes = np.diff(seg_offsets)
    by_size = np.argsort(-sizes, kind="stable")
    return np.concatenate([by_size[t::n_threads] for t in range(n_threads)]).astype(np.int64)


@njit(cache=True)
def time_positions(uid_t, seg_offsets):
    """For failures in time order: their index in the user-grouped array (core's layout).

    The r-th failure of user u (in time order) sits at seg_offsets[u] + r, so counts written
    by the time split land exactly where core.find_episodes expects them."""
    pos = np.empty(uid_t.shape[0], dtype=np.int64)
    seen = np.zeros(seg_offsets.shape[0] - 1, dtype=np.int64)
    for k in range(uid_t.shape[0]):
        u = uid_t[k]
        pos[k] = seg_offsets[u] + seen[u]
        seen[u] += 1
    return pos


def time_cuts(ts_t: np.ndarray, n_chunks: int) -> np.ndarray:
    """Cut the time-ordered failures into n_chunks of about equal size. A cut is moved
    forward until the timestamp changes, so equal timestamps never end up in two chunks."""
    F = ts_t.shape[0]
    cuts = [0]
    for c in range(1, n_chunks):
        k = max(c * F // n_chunks, cuts[-1])
        while 0 < k < F and ts_t[k] == ts_t[k - 1]:
            k += 1
        cuts.append(k)
    cuts.append(F)
    return np.array(cuts, dtype=np.int64)


# ----------------------------------------------------------------------------- compute (time split)


@njit(parallel=True, cache=True)
def time_split_counts(ts_t, uid_t, pos_t, cuts, window_s, j_out, count_out, halo_out):
    """Variant B: chunk c OWNS the time-ordered failures [cuts[c], cuts[c+1]) and also READS
    the halo before it: failures with ts > ts(first owned) - W. The halo is just an index
    range in shared memory - nothing is copied between threads.

    Inside its range a chunk sorts the failures by (user, time) and counts each owned
    failure's window with a binary search, exactly as core.window_counts does. It writes
    j and count at the failure's place in core's user-grouped layout (pos_t), so each value
    is written exactly once, by the chunk that owns that failure.
    """
    for c in prange(cuts.shape[0] - 1):
        a, b = cuts[c], cuts[c + 1]
        if a == b:
            halo_out[c] = 0
            continue
        h = core.lower_bound(ts_t, 0, a, ts_t[a] - window_s + 1)  # first failure of the halo
        halo_out[c] = a - h
        base = ts_t[h]
        key = uid_t[h:b].astype(np.int64) * (1 << 32) + (ts_t[h:b] - base)
        local = np.argsort(key, kind="mergesort")  # stable: ties keep time/file order
        lts = ts_t[h:b][local]
        luid = uid_t[h:b][local]
        seg_start = 0
        for p in range(b - h):
            if p == 0 or luid[p] != luid[p - 1]:
                seg_start = p
            k = h + local[p]
            if k < a:
                continue  # halo failure: read only, its owner is the previous chunk
            q = core.lower_bound(lts, seg_start, p + 1, lts[p] - window_s + 1)
            g = pos_t[k]
            count_out[g] = p - q + 1
            j_out[g] = pos_t[h + local[q]]


# ----------------------------------------------------------------------------- backend entry

_compile_s: float | None = None


def warm_up() -> float:
    """Compile every function once on a tiny input (only the first call does the work)."""
    global _compile_s
    if _compile_s is None:
        start = time.perf_counter()
        tiny = EventArrays(
            ts=np.array([0, 0, 1, 5], dtype=np.int64),
            user_id=np.array([0, 1, 0, 0], dtype=np.int32),
            ip_id=np.zeros(4, dtype=np.int32),
            is_fail=np.array([1, 1, 1, 0], dtype=np.uint8),
            users=["a", "b"],
        )
        core.warm_up()
        for split in ("users", "time"):
            _run(tiny, RuleParams(60, 2), 2, split, True)
        _compile_s = time.perf_counter() - start
    return _compile_s


def detect(
    events: EventArrays,
    params: RuleParams,
    threads: int | None = None,
    split: str = "users",
    balance: bool = True,
    **_: object,
) -> DetectionResult:
    """Detect with `threads` OpenMP threads (default: all). threads=1 is the sequential
    baseline for speed-up: the same code, so the comparison is fair."""
    if split not in ("users", "time"):
        raise ValueError("split must be 'users' or 'time'")
    max_threads = numba.config.NUMBA_NUM_THREADS
    threads = threads or max_threads
    if not 1 <= threads <= max_threads:
        raise ValueError(f"threads must be between 1 and {max_threads}")

    compile_s = warm_up()
    numba.set_num_threads(threads)
    result = _run(events, params, threads, split, balance)
    result.timings["compile"] = compile_s
    result.info.update({"threads": threads, "split": split, "balance": balance,
                        "threading_layer": numba.threading_layer()})
    return result


def _run(events: EventArrays, params: RuleParams, T: int, split: str, balance: bool) -> DetectionResult:
    timer = PhaseTimer()
    U = len(events.users)
    info: dict = {}
    with timer.phase("total"):
        with timer.phase("prepare"):
            ts_sorted, seg_offsets, n_events, n_fail, first, last = group_by_user(
                events.user_id, events.ts, events.is_fail, U, T)
            if split == "users":
                order = balanced_order(seg_offsets, T) if balance else np.arange(U, dtype=np.int64)
            else:
                mask = events.is_fail.astype(bool)
                ts_f, uid_f = events.ts[mask], events.user_id[mask]
                by_time = np.argsort(ts_f, kind="stable")  # ties keep file order
                ts_t, uid_t = ts_f[by_time].astype(np.int64), uid_f[by_time].astype(np.int64)
                pos_t = time_positions(uid_t, seg_offsets)
                cuts = time_cuts(ts_t, T)

        with timer.phase("compute"):
            if split == "users":
                j, count = core.window_counts(ts_sorted, seg_offsets, params.window_s, order)
            else:
                F = ts_sorted.shape[0]
                j = np.empty(F, dtype=np.int64)
                count = np.empty(F, dtype=np.int64)
                halo = np.zeros(cuts.shape[0] - 1, dtype=np.int64)
                time_split_counts(ts_t, uid_t, pos_t, cuts, params.window_s, j, count, halo)
                info["halo_events"] = int(halo.sum())  # failures read twice because of the halo
                info["chunks"] = int(cuts.shape[0] - 1)

        with timer.phase("merge"):
            ep_user, first_j, first_i, last_i, max_count = core.find_episodes(
                seg_offsets, j, count, params.threshold)
            alarms = core.alarms_frame(ts_sorted, ep_user, first_j, first_i, last_i, events.users)
            summary = core.summary_frame(events.users, n_events, n_fail, first, last,
                                         max_count, ep_user)
    return DetectionResult(alarms, summary, timer.as_dict(), info)
