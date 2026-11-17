"""MPI backend: mpi4py + MS-MPI, accounts split between processes (milestone M4).

Design: docs/ARCHITECTURE.md section 7.2. Run with:
    mpiexec -n 4 python -m login_correlation detect --backend mpi --data data\\demo --out results\\mpi

How it works:

    rank 0   partition()   every account gets ONE owner: owner_of(user, P) = crc32(user) % P.
                           A one-pass counting sort (Numba) reorders the events so each rank's
                           events are one block (stable: file order is kept inside a block,
                           so equal timestamps stay in order).
                           Each rank's accounts get local ids 0..U_r-1.
    all      scatter       comm.scatter  -> each rank gets its account names and event count
                           comm.Scatterv -> each rank gets its block of ts, user_id, is_fail
                                            (NumPy buffers, no pickling: the fast way)
    all      compute       core.detect on the local accounts only - the same, already tested
                           algorithm as everywhere else. No rank needs data of another rank,
                           because all events of an account live on its owner.
    all      gather        comm.gather -> rank 0 receives every rank's alarms and summary rows
    rank 0   merge()       concatenate + normalize -> the same files as the reference backend

Every account lives on exactly one rank, so every alarm has exactly one owner: no alarm can be
reported twice or split between two ranks. This is the task's "each alarm has one owner".

Why crc32 and not Python's hash(): hash() of a string is randomised per process, so two
ranks would disagree about who owns an account. crc32 gives the same number everywhere.

Timing (MPI.Wtime, reported on rank 0):
    partition  rank 0 builds the blocks          scatter  sending the blocks
    compute    slowest rank's local detection    gather   collecting + merging the results
    local_min / local_max show how evenly the work was spread (load balance).

partition() and merge() are plain functions without MPI, so tests can check them in one
process: partition the data, run core.detect on every part, merge -> must equal reference.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from login_correlation import core
from login_correlation.common import (
    DetectionResult,
    EventArrays,
    RuleParams,
    normalize_alarms,
    normalize_summary,
)


def owner_of(user: str, size: int) -> int:
    """Rank that owns `user`. crc32 instead of hash(): hash() differs between processes."""
    return zlib.crc32(user.encode("utf-8")) % size


# ----------------------------------------------------------------------------- without MPI


@dataclass
class Partition:
    """The data of all ranks, prepared on rank 0. Rank r's events are
    ts[displs[r] : displs[r] + counts[r]] (same for user_id, is_fail)."""

    ts: np.ndarray            # int64, events grouped by owner rank
    user_id: np.ndarray       # int32, LOCAL account id inside the owner rank
    is_fail: np.ndarray       # uint8
    counts: np.ndarray        # int64[P] events per rank
    displs: np.ndarray        # int64[P] where each rank's block starts
    users: list[list[str]]    # users[r] = account names owned by rank r (local id = position)

    def part(self, r: int) -> EventArrays:
        """Rank r's share as EventArrays (what rank r has after the scatter)."""
        a, b = self.displs[r], self.displs[r] + self.counts[r]
        return EventArrays(ts=self.ts[a:b].copy(), user_id=self.user_id[a:b].copy(),
                           ip_id=np.zeros(b - a, dtype=np.int32),
                           is_fail=self.is_fail[a:b].copy(), users=self.users[r])


def partition(events: EventArrays, size: int) -> Partition:
    """Split the events between `size` ranks by account owner (runs on rank 0)."""
    U = len(events.users)
    owner = np.fromiter((owner_of(u, size) for u in events.users), dtype=np.int64, count=U)
    # local id of every account inside its owner rank: its position among that rank's accounts
    local_id = np.zeros(U, dtype=np.int32)
    users: list[list[str]] = [[] for _ in range(size)]
    for g, (name, r) in enumerate(zip(events.users, owner.tolist())):
        local_id[g] = len(users[r])
        users[r].append(name)

    ts, uid, fail, counts, displs = _group_by_owner(
        events.ts, events.user_id, events.is_fail, owner, local_id, size)
    return Partition(ts=ts, user_id=uid, is_fail=fail, counts=counts, displs=displs, users=users)


@njit(cache=True)
def _group_by_owner(ts, user_id, is_fail, owner, local_id, size):
    """Counting sort of the events by owner rank, in one pass (compiled with Numba).

    First count the events of each rank, then walk the events once and write each one
    straight to the next free place in its rank's block. Walking in file order keeps the
    file order inside every block (the sort is stable), which the rule needs for events
    with equal timestamps. Measured on 10 M events: ~0.1 s, versus ~0.7-1.3 s for numpy's
    argsort + three fancy-indexing copies (each copy allocates another 10 M array).
    """
    n = ts.shape[0]
    counts = np.zeros(size, dtype=np.int64)
    for k in range(n):
        counts[owner[user_id[k]]] += 1
    displs = np.zeros(size, dtype=np.int64)
    for r in range(1, size):
        displs[r] = displs[r - 1] + counts[r - 1]
    out_ts = np.empty(n, dtype=np.int64)
    out_uid = np.empty(n, dtype=np.int32)
    out_fail = np.empty(n, dtype=np.uint8)
    nxt = displs.copy()
    for k in range(n):
        g = user_id[k]
        w = nxt[owner[g]]
        out_ts[w] = ts[k]
        out_uid[w] = local_id[g]
        out_fail[w] = is_fail[k]
        nxt[owner[g]] = w + 1
    return out_ts, out_uid, out_fail, counts, displs


def merge(parts: list[tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine the (alarms, summary) of all ranks into one result (runs on rank 0)."""
    alarms = pd.concat([a for a, _ in parts], ignore_index=True)
    summary = pd.concat([s for _, s in parts], ignore_index=True)
    return normalize_alarms(alarms), normalize_summary(summary)


# ----------------------------------------------------------------------------- with MPI


def detect(events: EventArrays | None, params: RuleParams, threads: int | None = None,
           **_: object) -> DetectionResult | None:
    """Run on every rank (mpiexec). `events` is given only on rank 0; other ranks pass None.
    Returns the result on rank 0 and None on the other ranks.

    threads = Numba threads per rank (default 1 = pure MPI, one core per process)."""
    import numba
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    numba.set_num_threads(max(1, min(threads or 1, numba.config.NUMBA_NUM_THREADS)))
    compile_s = core.warm_up()  # compile (or load from cache) before any timing
    if rank == 0:  # compile the partition step too, so it is not counted as work
        _group_by_owner(np.zeros(1, np.int64), np.zeros(1, np.int32), np.zeros(1, np.uint8),
                        np.zeros(1, np.int64), np.zeros(1, np.int32), 1)
    comm.Barrier()
    t0 = MPI.Wtime()

    # 1. partition on rank 0
    if rank == 0:
        p = partition(events, size)
        meta = [(int(p.counts[r]), p.users[r]) for r in range(size)]
    else:
        p, meta = None, None
    t_part = MPI.Wtime() - t0

    # 2. scatter: account names (small, pickled), then the event arrays (NumPy buffers)
    t1 = MPI.Wtime()
    n_local, my_users = comm.scatter(meta, root=0)
    ts = np.empty(n_local, dtype=np.int64)
    uid = np.empty(n_local, dtype=np.int32)
    fail = np.empty(n_local, dtype=np.uint8)
    for send, recv, mpi_type in ((p.ts if p else None, ts, MPI.INT64_T),
                                 (p.user_id if p else None, uid, MPI.INT32_T),
                                 (p.is_fail if p else None, fail, MPI.UINT8_T)):
        sendbuf = [send, p.counts, p.displs, mpi_type] if rank == 0 else None
        comm.Scatterv(sendbuf, recv, root=0)
    t_scatter = MPI.Wtime() - t1

    # 3. local detection - each rank only has its own accounts
    t2 = MPI.Wtime()
    local = core.detect(EventArrays(ts=ts, user_id=uid, ip_id=np.zeros(n_local, dtype=np.int32),
                                    is_fail=fail, users=my_users), params)
    t_local = MPI.Wtime() - t2
    local_times = comm.gather(t_local, root=0)

    # 4. gather the results on rank 0 and merge them
    t3 = MPI.Wtime()
    parts = comm.gather((local.alarms, local.summary), root=0)
    if rank != 0:
        return None
    alarms, summary = merge(parts)
    t_gather = MPI.Wtime() - t3
    total = MPI.Wtime() - t0

    timings = {
        "prepare": t_part + t_scatter,
        "compute": max(local_times),
        "collect": t_gather,
        "total": total,
        "partition": t_part,
        "scatter": t_scatter,
        "gather": t_gather,
        "local_min": min(local_times),
        "local_max": max(local_times),
        "compile": compile_s,
    }
    info = {
        "ranks": size,
        "threads_per_rank": numba.get_num_threads(),
        "events_per_rank": [int(c) for c in p.counts],
        "accounts_per_rank": [len(u) for u in p.users],
        "mpi_library": MPI.Get_library_version().splitlines()[0].strip(),
    }
    return DetectionResult(alarms, summary, timings, info)
