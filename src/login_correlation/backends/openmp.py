"""OpenMP backend: Numba `prange` running on the OpenMP threading layer. TODO M3.

Design: docs/ARCHITECTURE.md section 7.1. The task names two ways to split the work,
and both are implemented, chosen with `split`:

split="users"  (variant A - accounts between threads)
    prepare : mask = is_fail == 1
              order = np.lexsort((ts[mask], user_id[mask]))  # stable -> ties keep file order
              seg_offsets = concatenate([0], cumsum(bincount(user_id[mask], minlength=U)))
              users interleaved by segment size (load balancing)
    compute : core.window_counts(...)  - prange over users, runs on OpenMP threads

split="time"   (variant B - groups of time-ordered events, with a halo)
    prepare : failures sorted by (ts, seq); cut into `threads` chunks of about equal size.
              A cut is moved forward until the timestamp changes, so equal timestamps
              are never split between two chunks.
    compute : prange over chunks. Chunk k owns the failures with ts in [t_k, t_{k+1}) and
              ALSO reads the halo: failures with ts in (t_k - W, t_k), found with
              core.lower_bound. Only the owned failures get a count, so each count is
              written exactly once. The halo is just an index range in shared memory;
              nothing is copied.

both    merge   : core.merge_episodes(...) + core.build_summary(...) on the counts, which are
                  in the original order, so an alarm that crosses a chunk border is merged
                  normally.

Call the kernel once on a tiny input before timing (JIT warm-up) and report the compile
time separately. threads=1 is the sequential baseline for speed-up.
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")  # must be set before Numba starts its threads

from login_correlation.common import DetectionResult, EventArrays, RuleParams  # noqa: E402


def detect(events: EventArrays, params: RuleParams, threads: int | None = None,
           split: str = "users", **_: object) -> DetectionResult:
    """TODO(M3): numba.set_num_threads(threads), then variant `split` ("users" or "time")."""
    if split not in ("users", "time"):
        raise ValueError("split must be 'users' or 'time'")
    raise NotImplementedError("openmp backend - milestone M3")
