"""MPI backend: mpi4py + MS-MPI, accounts split between processes. TODO M4.

Design: docs/ARCHITECTURE.md section 7.2. Run with:
    mpiexec -n 4 python -m login_correlation detect --backend mpi --data data\\demo --out results\\mpi

Steps:
    rank 0 : events are loaded by __main__ on rank 0 only
             owner of a user = owner_of(user, size)
             sort rows by owner, count rows per rank
    all    : comm.Scatterv(ts, user_id, is_fail)  + comm.bcast(users)
    all    : core.window_counts + core.merge_episodes on the local users (1 thread)
    all    : comm.gather(alarm rows, summary rows) to rank 0
    rank 0 : concatenate + normalize, return DetectionResult; other ranks return None

A user lives on exactly one rank, so every alarm has exactly one owner.
Time scatter / compute / gather with MPI.Wtime() and report the maximum over ranks.
"""

from __future__ import annotations

import zlib

from login_correlation.common import DetectionResult, EventArrays, RuleParams


def owner_of(user: str, size: int) -> int:
    """Rank that owns `user`. crc32 instead of hash(): hash() differs between processes."""
    return zlib.crc32(user.encode("utf-8")) % size


def detect(events: EventArrays | None, params: RuleParams, **_: object) -> DetectionResult | None:
    """`events` is None on ranks other than 0. TODO(M4)."""
    raise NotImplementedError("mpi backend - milestone M4")
