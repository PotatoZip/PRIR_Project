"""OpenCL backend (backends/opencl.py, milestone M5).

The GPU kernels are checked one by one against the CPU code (core.py), then the whole
backend against the reference, including the hard cases. All tests are skipped when
PyOpenCL or an OpenCL device is missing.
"""

from pathlib import Path

import numpy as np
import pytest
from conftest import random_events

pytest.importorskip("pyopencl")
pytest.importorskip("mako")

from login_correlation import core  # noqa: E402
from login_correlation.__main__ import main  # noqa: E402
from login_correlation.backends import opencl, reference  # noqa: E402
from login_correlation.common import EventArrays, RuleParams  # noqa: E402

try:
    GPU = opencl._gpu(None)
except Exception as exc:  # noqa: BLE001 - no driver / no device
    pytest.skip(f"no usable OpenCL device: {exc}", allow_module_level=True)

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


def same(a, b):
    return a.alarms.equals(b.alarms) and a.summary.equals(b.summary)


# ----------------------------------------------------------------- device choice


def test_default_device_is_a_native_gpu():
    dev = opencl.choose_device(None)
    assert "OpenCLOn12" not in dev.platform.name


def test_explicit_device_and_bad_device():
    assert opencl.choose_device("0:0") is not None
    with pytest.raises(ValueError):
        opencl.choose_device("99:0")
    with pytest.raises(ValueError):
        opencl.choose_device("gpu")


# ----------------------------------------------------------------- kernels = CPU code


def test_window_kernel_equals_core():
    """window_count on the GPU must give the same j and count as core.window_counts."""
    import pyopencl.array as cla

    ev = random_events(5)
    ts_sorted, off = core.sort_failures(ev)
    W = 60
    j_cpu, c_cpu = core.window_counts(ts_sorted, off, W, np.arange(len(ev.users), dtype=np.int64))
    # build the GPU keys from the same sorted data: user id in the high 32 bits
    uid = np.repeat(np.arange(len(ev.users)), np.diff(off)).astype(np.uint64)
    keys = (uid << np.uint64(32)) | (ts_sorted - ts_sorted.min()).astype(np.uint64)
    q = GPU.queue
    F = len(keys)
    d_keys = cla.to_device(q, keys)
    d_j = cla.empty(q, F, dtype=np.uint32)
    d_c = cla.empty(q, F, dtype=np.uint32)
    d_max = cla.zeros(q, len(ev.users), dtype=np.int32)
    GPU.k_window(q, (F,), None, d_keys.data, np.uint32(F), np.uint32(W), d_j.data, d_c.data, d_max.data)
    assert np.array_equal(d_j.get(), j_cpu)
    assert np.array_equal(d_c.get(), c_cpu)


def test_gpu_radix_sort_equals_numpy():
    import pyopencl.array as cla

    rng = np.random.default_rng(1)
    keys = (rng.integers(0, 1000, 50_000).astype(np.uint64) << np.uint64(32)) | rng.integers(
        0, 3600, 50_000).astype(np.uint64)
    (d_sorted,), _ = GPU.sorter(cla.to_device(GPU.queue, keys), key_bits=42, queue=GPU.queue)
    assert np.array_equal(d_sorted.get(), np.sort(keys))


@pytest.mark.parametrize("seed", [0, 1])
def test_gpu_user_stats_equal_core(seed):
    ev = random_events(seed)
    res = opencl.detect(ev, RuleParams(60, 5))
    exp = reference.detect(ev, RuleParams(60, 5))
    cols = ["user", "n_events", "n_fail", "n_success", "first_ts", "last_ts", "max_window_count"]
    assert res.summary[cols].equals(exp.summary[cols])


# ----------------------------------------------------------------- whole backend = reference


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("params", [RuleParams(60, 5), RuleParams(10, 2), RuleParams(300, 10),
                                    RuleParams(1, 1), RuleParams(3600, 50)])
def test_opencl_equals_reference(seed, params):
    ev = random_events(seed)
    assert same(opencl.detect(ev, params), reference.detect(ev, params))


def test_window_edges_and_equal_timestamps():
    rows = [(t, "edge", "ip", "FAIL") for t in (100, 130, 160)]      # 60 s apart: no alarm (N=3)
    rows += [(t, "inside", "ip", "FAIL") for t in (100, 130, 159)]   # 59 s: alarm
    rows += [(500, "same", "ip", "FAIL")] * 3                         # same second: alarm
    rows += [(t, "", "ip", "FAIL") for t in (10, 11, 12)]             # empty user field
    ev = EventArrays.from_rows(rows, registry=["nobody"])
    p = RuleParams(60, 3)
    res = opencl.detect(ev, p)
    assert same(res, reference.detect(ev, p))
    assert set(res.alarms["user"]) == {"inside", "same", "<empty>"}


def test_no_failures_and_no_events():
    p = RuleParams(60, 1)
    ev = EventArrays.from_rows([(1, "a", "ip", "SUCCESS")], registry=["a", "b"])
    assert same(opencl.detect(ev, p), reference.detect(ev, p))
    empty = EventArrays.from_rows([], registry=["a"])
    assert same(opencl.detect(empty, p), reference.detect(empty, p))


def test_timestamps_far_from_zero():
    """ts is sent as a 32-bit offset from the first event; real epoch values must survive."""
    base = 1_767_225_600
    ev = EventArrays.from_rows([(base + t, "u", "ip", "FAIL") for t in (0, 5, 10)])
    res = opencl.detect(ev, RuleParams(60, 3))
    assert res.alarms.iloc[0]["start_ts"] == base and res.alarms.iloc[0]["end_ts"] == base + 10


def test_too_long_time_span_is_rejected():
    ev = EventArrays.from_rows([(0, "u", "ip", "FAIL"), (2**32 + 5, "u", "ip", "FAIL")])
    with pytest.raises(ValueError):
        opencl.detect(ev, RuleParams(60, 2))


# ----------------------------------------------------------------- reporting and command line


def test_every_phase_is_timed_separately():
    res = opencl.detect(random_events(0), RuleParams(60, 5))
    for phase in ("gpu_prepare", "h2d", "filter", "sort", "count", "stats", "d2h", "merge",
                  "prepare", "compute", "collect", "total", "compile"):
        assert phase in res.timings
    assert res.info["device"] and res.info["failures"] > 0


def test_cli_opencl_equals_reference(tmp_path):
    ref, gpu = tmp_path / "ref", tmp_path / "gpu"
    assert main(["detect", "--data", str(SAMPLE), "--out", str(ref)]) == 0
    assert main(["detect", "--backend", "opencl", "--device", "0:0", "--data", str(SAMPLE),
                 "--out", str(gpu)]) == 0
    assert main(["compare", str(ref), str(gpu)]) == 0


def test_bench_opencl_phases_on_sample(tmp_path):
    """The benchmark code (bench.py) runs end to end and draws the phase chart."""
    from login_correlation import bench

    runs = bench.opencl_phases(SAMPLE, repeats=2)
    assert len(runs) == 2
    assert {"load", "encode", *bench.GPU_PHASES, "gpu_total"} <= set(runs.columns)
    bench.plot_opencl(runs, tmp_path / "gpu.png")
    assert (tmp_path / "gpu.png").stat().st_size > 0
