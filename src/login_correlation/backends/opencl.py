"""OpenCL backend: PyOpenCL on the Intel Iris Xe GPU (milestone M5).

Design: docs/ARCHITECTURE.md section 7.3. Run with:
    python -m login_correlation detect --backend opencl --device 0:0 --data data\\demo --out results\\gpu

The task: "after encoding the text fields, filter, sort and count the events on the GPU;
show the cost of encoding and data preparation separately". The phases, each timed on its own
(wall clock, after queue.finish(), so a phase's time includes its GPU work):

    encode       CPU  text -> numbers (user name -> id, result -> 0/1); done by common.encode,
                      measured by the command line as "encode"
    gpu_prepare  CPU  ts -> uint32 offset from the first event (ts_rel); arrays made contiguous
    h2d               copy ts_rel, user_id, is_fail to the GPU
    filter       GPU  key = (user_id << 32) | ts_rel for every event, then copy_if(is_fail)
                      keeps only the failures' keys (stream compaction)
    sort         GPU  RadixSort of the 64-bit keys. Sorting one combined key = sorting by
                      (user, time). Only the bits that are really used are sorted.
    count        GPU  window_count kernel: one work-item per failure, binary search for the
                      first key >= key - W + 1. Because the user id is in the high bits, the
                      search can never reach another user's failures. Also atomic_max of the
                      count per user (max_window_count for the summary).
    stats        GPU  user_stats kernel: one work-item per event, atomic_inc / atomic_min /
                      atomic_max -> n_events, n_fail, first_ts, last_ts per user
    d2h               copy the sorted keys, j, count and the per-user arrays back
    merge        CPU  core.find_episodes (triggers -> alarms) + output tables, the same code
                      as every other backend

Kernels and the radix sorter are built once per OpenCL context and cached; that build time
is reported as "compile" (not part of "total").
"""

from __future__ import annotations

import os
import time

import numpy as np

from login_correlation import core
from login_correlation.common import DetectionResult, EventArrays, PhaseTimer, RuleParams

KERNELS = r"""
// One work-item per failure. keys are sorted; key = (user << 32) | ts_rel.
__kernel void window_count(__global const ulong *keys, const uint n, const uint window_s,
                           __global uint *j_out, __global uint *count_out,
                           __global int *max_count)
{
    uint i = get_global_id(0);
    if (i >= n) return;
    ulong key = keys[i];
    uint ts = (uint)(key & 0xFFFFFFFFUL);
    ulong user_base = key & 0xFFFFFFFF00000000UL;
    // first key inside the window (t - W, t]; clamp at the user's time 0
    ulong target = user_base | (ts >= window_s - 1 ? (ulong)(ts - window_s + 1) : 0UL);

    uint lo = 0, hi = i;                 // lower_bound in keys[0 .. i]
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2;
        if (keys[mid] < target) lo = mid + 1; else hi = mid;
    }
    uint c = i - lo + 1;
    j_out[i] = lo;
    count_out[i] = c;
    atomic_max(&max_count[(uint)(key >> 32)], (int)c);
}

// One work-item per event: per-user statistics with atomic operations.
__kernel void user_stats(__global const uint *uid, __global const uint *ts,
                         __global const uchar *fail, const uint n,
                         __global int *n_events, __global int *n_fail,
                         __global uint *first, __global uint *last)
{
    uint i = get_global_id(0);
    if (i >= n) return;
    uint u = uid[i];
    atomic_inc(&n_events[u]);
    if (fail[i]) atomic_inc(&n_fail[u]);
    atomic_min(&first[u], ts[i]);
    atomic_max(&last[u], ts[i]);
}
"""


# ----------------------------------------------------------------------------- device and kernels


def choose_device(device: str | None = None):
    """Return a pyopencl Device.

    device = "P:D" (platform P, device D, as listed by `python -m login_correlation check`).
    Default: the first GPU of a native platform. Microsoft's "OpenCLOn12" layer (OpenCL on top
    of DirectX 12) is skipped unless asked for, because it is slower and less complete.
    """
    import pyopencl as cl

    platforms = cl.get_platforms()
    device = device or os.environ.get("PYOPENCL_CTX") or None
    if device:
        try:
            p, d = (int(x) for x in device.split(":"))
            return platforms[p].get_devices()[d]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"no OpenCL device {device!r} (use P:D, see `python -m login_correlation check`)") from exc
    candidates = [(p, d) for p in platforms for d in p.get_devices()]
    for prefer_gpu in (True, False):
        for p, d in candidates:
            if "OpenCLOn12" in p.name:
                continue
            if not prefer_gpu or d.type & cl.device_type.GPU:
                return d
    if candidates:
        return candidates[0][1]
    raise RuntimeError("no OpenCL device found - update the graphics driver")


class _Gpu:
    """Context, queue, compiled kernels and radix sorter for one device (built once)."""

    def __init__(self, dev):
        import pyopencl as cl
        from pyopencl.algorithm import RadixSort
        from pyopencl.elementwise import ElementwiseKernel

        start = time.perf_counter()
        self.device = dev
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, KERNELS).build()
        # take each kernel out of the program ONCE and reuse it (getting it again on every
        # call would create a new kernel object each time)
        self.k_window = cl.Kernel(self.prg, "window_count")
        self.k_stats = cl.Kernel(self.prg, "user_stats")
        self.make_keys = ElementwiseKernel(
            self.ctx, "const uint *uid, const uint *ts, ulong *key",
            "key[i] = ((ulong)uid[i] << 32) | (ulong)ts[i]", "make_keys")
        self.sorter = RadixSort(self.ctx, "ulong *keys", key_expr="keys[i]",
                                sort_arg_names=["keys"], key_dtype=np.uint64)
        self.build_s = time.perf_counter() - start
        self.warm_s = 0.0


_cache: dict[str, _Gpu] = {}


def _gpu(device: str | None) -> _Gpu:
    dev = choose_device(device)
    key = f"{dev.platform.name}|{dev.name}"
    if key not in _cache:
        g = _Gpu(dev)
        _cache[key] = g
        # first run builds the copy_if and radix-sort kernels for these sizes/types
        t = time.perf_counter()
        tiny = EventArrays(ts=np.array([0, 1, 1, 70], dtype=np.int64),
                           user_id=np.array([0, 1, 0, 0], dtype=np.int32),
                           ip_id=np.zeros(4, dtype=np.int32),
                           is_fail=np.array([1, 1, 1, 0], dtype=np.uint8), users=["a", "b"])
        _run(g, tiny, RuleParams(60, 2))
        g.warm_s = time.perf_counter() - t
    return _cache[key]


# ----------------------------------------------------------------------------- detection


def detect(events: EventArrays, params: RuleParams, device: str | None = None,
           **_: object) -> DetectionResult:
    """Detect on the OpenCL device `device` ("P:D"; default: first native GPU)."""
    g = _gpu(device)
    result = _run(g, events, params)
    result.timings["compile"] = g.build_s + g.warm_s
    result.info.update({
        "device": g.device.name.strip(),
        "platform": g.device.platform.name.strip(),
        "compute_units": int(g.device.max_compute_units),
    })
    return result


def _run(g: _Gpu, events: EventArrays, params: RuleParams) -> DetectionResult:
    import pyopencl.array as cla
    from pyopencl.algorithm import copy_if

    q = g.queue
    timer = PhaseTimer()
    n = len(events)
    U = len(events.users)
    info: dict = {}

    with timer.phase("total"):
        # ---- CPU: last preparation step before the copy (the text was encoded already)
        with timer.phase("gpu_prepare"):
            ts_min = int(events.ts.min()) if n else 0
            span = int(events.ts.max()) - ts_min if n else 0
            if span >= 2**32:
                raise ValueError("the data spans more than 136 years; ts_rel does not fit in 32 bits")
            ts_rel = (events.ts - ts_min).astype(np.uint32)
            uid = np.ascontiguousarray(events.user_id).view(np.uint32)
            fail = np.ascontiguousarray(events.is_fail, dtype=np.uint8)
            key_bits = 32 + max(1, int(U).bit_length())  # only the bits that are used

        # ---- host -> device
        with timer.phase("h2d"):
            if n:
                d_ts = cla.to_device(q, ts_rel)
                d_uid = cla.to_device(q, uid)
                d_fail = cla.to_device(q, fail)
            q.finish()

        F = 0
        with timer.phase("filter"):
            if n:
                d_keys_all = cla.empty(q, n, dtype=np.uint64)
                g.make_keys(d_uid, d_ts, d_keys_all, queue=q)
                d_kept, d_count, _ = copy_if(d_keys_all, "fail[i] != 0",
                                             extra_args=[("fail", d_fail)], queue=q)
                F = int(d_count.get())
                d_keys = d_kept[:F]
            q.finish()

        with timer.phase("sort"):
            if F:
                (d_sorted,), _ = g.sorter(d_keys, key_bits=key_bits, queue=q)
            q.finish()

        d_max = cla.zeros(q, max(U, 1), dtype=np.int32)
        with timer.phase("count"):
            if F:
                d_j = cla.empty(q, F, dtype=np.uint32)
                d_c = cla.empty(q, F, dtype=np.uint32)
                g.k_window(q, (F,), None, d_sorted.data, np.uint32(F),
                                   np.uint32(params.window_s), d_j.data, d_c.data, d_max.data)
            q.finish()

        with timer.phase("stats"):
            d_nev = cla.zeros(q, max(U, 1), dtype=np.int32)
            d_nfail = cla.zeros(q, max(U, 1), dtype=np.int32)
            d_first = cla.empty(q, max(U, 1), dtype=np.uint32).fill(np.uint32(0xFFFFFFFF))
            d_last = cla.zeros(q, max(U, 1), dtype=np.uint32)
            if n:
                g.k_stats(q, (n,), None, d_uid.data, d_ts.data, d_fail.data, np.uint32(n),
                                 d_nev.data, d_nfail.data, d_first.data, d_last.data)
            q.finish()

        # ---- device -> host
        with timer.phase("d2h"):
            keys = d_sorted.get() if F else np.zeros(0, dtype=np.uint64)
            j = d_j.get().astype(np.int64) if F else np.zeros(0, dtype=np.int64)
            count = d_c.get().astype(np.int64) if F else np.zeros(0, dtype=np.int64)
            n_events = d_nev.get()[:U].astype(np.int64)
            n_fail = d_nfail.get()[:U].astype(np.int64)
            first = d_first.get()[:U].astype(np.int64) + ts_min
            last = d_last.get()[:U].astype(np.int64) + ts_min
            max_count = d_max.get()[:U].astype(np.int64)

        # ---- CPU: alarms and output tables (the same code as all other backends)
        with timer.phase("merge"):
            ts_sorted = (keys & np.uint64(0xFFFFFFFF)).astype(np.int64) + ts_min
            seg_offsets = np.zeros(U + 1, dtype=np.int64)
            np.cumsum(n_fail, out=seg_offsets[1:])
            ep_user, first_j, first_i, last_i, _ = core.find_episodes(
                seg_offsets, j, count, params.threshold)
            alarms = core.alarms_frame(ts_sorted, ep_user, first_j, first_i, last_i, events.users)
            first[n_events == 0] = 0
            last[n_events == 0] = 0
            summary = core.summary_frame(events.users, n_events, n_fail, first, last,
                                         max_count, ep_user)

    t = timer.as_dict()
    # the standard phase names used by all backends (docs/ARCHITECTURE.md section 8.3)
    t["prepare"] = t["gpu_prepare"] + t["h2d"]
    t["compute"] = t["filter"] + t["sort"] + t["count"] + t["stats"]
    t["collect"] = t["d2h"]
    info.update({"failures": F, "key_bits": key_bits})
    return DetectionResult(alarms, summary, t, info)
