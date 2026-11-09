"""OpenCL backend: PyOpenCL on the Intel Iris Xe GPU (device "0:0"). TODO M5.

Design: docs/ARCHITECTURE.md section 7.3. Every phase is timed separately (the task asks
for the encoding cost to be shown apart from the GPU work):

    1 encode  CPU  common.encode + ts_rel = (ts - ts.min()).astype(uint32)
    2 h2d          pyopencl.array.to_device
    3 filter  GPU  pyopencl.algorithm.copy_if   (is_fail == 1)
    4 sort    GPU  pyopencl.algorithm.RadixSort on key = (uint64(user_id) << 32) | ts_rel
    5 count   GPU  WINDOW_KERNEL below
    6 hist    GPU  per-user counters with atomic_inc
    7 d2h          .get()
    8 merge   CPU  core.merge_episodes + core.build_summary

Per-kernel times: queue created with PROFILING_ENABLE, event.profile.end - event.profile.start (ns).
"""

from __future__ import annotations

from login_correlation.common import DetectionResult, EventArrays, RuleParams

# With key = user << 32 | ts, searching for key - W + 1 can never reach another user's
# events: all earlier users have smaller keys.
WINDOW_KERNEL = r"""
__kernel void window_count(__global const ulong *keys, const uint n,
                           const uint window_s, const uint threshold,
                           __global uint *j_out, __global uint *count_out,
                           __global uchar *trigger_out)
{
    uint i = get_global_id(0);
    if (i >= n) return;
    ulong key = keys[i];
    uint ts = (uint)(key & 0xFFFFFFFFUL);
    ulong target = (key & 0xFFFFFFFF00000000UL) | (ts >= window_s - 1 ? (ulong)(ts - window_s + 1) : 0UL);

    uint lo = i;   /* TODO(M5): binary search for the first lo in [0, i] with keys[lo] >= target */

    j_out[i] = lo;
    count_out[i] = i - lo + 1;
    trigger_out[i] = (i - lo + 1) >= threshold;
}
"""


def detect(events: EventArrays, params: RuleParams, device: str | None = None,
           **_: object) -> DetectionResult:
    """TODO(M5). device = "platform:device", default "0:0" (native Intel driver)."""
    raise NotImplementedError("opencl backend - milestone M5")
