"""OpenMP backend (backends/openmp.py, milestone M3).

tests/test_backends.py already checks openmp = reference on random data for several
thread counts and both splits. This file checks the parallel building blocks one by one,
and the hard cases for the time split: chunk borders inside an attack and equal timestamps.
"""

import numba
import numpy as np
import pytest
from conftest import random_events

from login_correlation import core
from login_correlation.__main__ import main
from login_correlation.backends import openmp, reference
from login_correlation.common import EventArrays, RuleParams

THREADS = [1, 2, 3, 7, 12]


def same(a, b):
    return a.alarms.equals(b.alarms) and a.summary.equals(b.summary)


# ----------------------------------------------------------------- building blocks


@pytest.mark.parametrize("threads", THREADS)
def test_parallel_grouping_equals_sequential(threads):
    """group_by_user must give core.sort_failures' layout and core's per-user statistics."""
    ev = random_events(1)
    U = len(ev.users)
    ts_a, off_a = core.sort_failures(ev)
    n_events, first, last = core._user_stats(ev.user_id, ev.ts, U)
    ts_b, off_b, ne, nf, fi, la = openmp.group_by_user(ev.user_id, ev.ts, ev.is_fail, U, threads)
    assert np.array_equal(off_a, off_b)
    assert np.array_equal(ts_a, ts_b)
    assert np.array_equal(ne, n_events)
    assert np.array_equal(nf, np.bincount(ev.user_id, weights=ev.is_fail, minlength=U))
    has = n_events > 0
    assert np.array_equal(fi[has], first[has]) and np.array_equal(la[has], last[has])


@pytest.mark.parametrize("threads", [1, 2, 5, 12])
def test_balanced_order_is_a_permutation_and_balances(threads):
    sizes = np.array([100, 1, 1, 1, 50, 50, 2, 2, 90, 3])
    off = np.concatenate([[0], np.cumsum(sizes)])
    order = openmp.balanced_order(off, threads)
    assert sorted(order.tolist()) == list(range(len(sizes)))
    if threads == 2:   # block 0 = biggest, 3rd, 5th...; block 1 = 2nd, 4th, ...
        half = len(order) // 2
        work = [sizes[order[:half]].sum(), sizes[order[half:]].sum()]
        assert abs(work[0] - work[1]) <= 100


def test_time_cuts_never_split_equal_timestamps():
    ts = np.array([1, 2, 2, 2, 2, 3, 4, 4, 5, 6], dtype=np.int64)
    for chunks in range(1, 11):
        cuts = openmp.time_cuts(ts, chunks)
        assert cuts[0] == 0 and cuts[-1] == len(ts) and len(cuts) == chunks + 1
        assert (np.diff(cuts) >= 0).all()
        for k in cuts[1:-1]:
            assert k in (0, len(ts)) or ts[k] != ts[k - 1]


def test_time_positions_match_user_grouping():
    ev = random_events(4)
    ts_sorted, off = core.sort_failures(ev)
    mask = ev.is_fail.astype(bool)
    by_time = np.argsort(ev.ts[mask], kind="stable")
    ts_t = ev.ts[mask][by_time]
    uid_t = ev.user_id[mask][by_time].astype(np.int64)
    pos = openmp.time_positions(uid_t, off)
    assert sorted(pos.tolist()) == list(range(len(ts_sorted)))      # every slot exactly once
    assert np.array_equal(ts_sorted[pos], ts_t)                      # and the right timestamp


# ----------------------------------------------------------------- time split: hard cases


@pytest.mark.parametrize("threads", THREADS)
def test_attack_cut_by_chunk_border_is_one_alarm(threads):
    # one attacker, 100 failures 1 s apart (one long attack) + background noise
    rows = [(t, "victim", "203.0.113.5", "FAIL") for t in range(1000, 1100)]
    rows += [(t, f"u{t % 7}", "10.0.0.1", "FAIL" if t % 3 else "SUCCESS") for t in range(0, 3000, 2)]
    ev = EventArrays.from_rows(rows)
    p = RuleParams(60, 5)
    got = openmp.detect(ev, p, threads=threads, split="time")
    assert same(got, reference.detect(ev, p))
    victim = got.alarms[got.alarms["user"] == "victim"]
    assert len(victim) == 1 and victim.iloc[0]["n_failures"] == 100
    if threads > 1:
        assert got.info["halo_events"] > 0      # chunks really read their halo


@pytest.mark.parametrize("threads", THREADS)
def test_all_failures_in_one_second(threads):
    ev = EventArrays.from_rows([(500, f"u{k % 3}", "ip", "FAIL") for k in range(30)])
    p = RuleParams(60, 5)
    assert same(openmp.detect(ev, p, threads=threads, split="time"), reference.detect(ev, p))


@pytest.mark.parametrize("split", ["users", "time"])
@pytest.mark.parametrize("threads", THREADS)
def test_more_threads_than_failures(split, threads):
    ev = EventArrays.from_rows([(1, "a", "ip", "FAIL"), (2, "a", "ip", "FAIL")], registry=["a", "b", "c"])
    p = RuleParams(60, 2)
    assert same(openmp.detect(ev, p, threads=threads, split=split), reference.detect(ev, p))


@pytest.mark.parametrize("split", ["users", "time"])
def test_no_failures(split):
    ev = EventArrays.from_rows([(1, "a", "ip", "SUCCESS")], registry=["a", "b"])
    p = RuleParams(60, 1)
    assert same(openmp.detect(ev, p, threads=4, split=split), reference.detect(ev, p))


@pytest.mark.parametrize("split", ["users", "time"])
@pytest.mark.parametrize("seed", range(4))
def test_both_splits_on_random_data_all_thread_counts(split, seed):
    ev = random_events(seed)
    p = RuleParams(30, 3)
    expected = reference.detect(ev, p)
    for threads in THREADS:
        assert same(openmp.detect(ev, p, threads=threads, split=split), expected)


# ----------------------------------------------------------------- settings and reporting


def test_runs_on_openmp_layer():
    res = openmp.detect(random_events(0), RuleParams(60, 5), threads=2)
    assert res.info["threading_layer"] == "omp"
    assert res.info["threads"] == 2
    assert {"prepare", "compute", "merge", "total", "compile"} <= set(res.timings)


def test_invalid_settings():
    ev = random_events(0)
    with pytest.raises(ValueError):
        openmp.detect(ev, RuleParams(60, 5), split="diagonal")
    with pytest.raises(ValueError):
        openmp.detect(ev, RuleParams(60, 5), threads=numba.config.NUMBA_NUM_THREADS + 1)


@pytest.mark.parametrize("split", ["users", "time"])
def test_cli_openmp_equals_reference(tmp_path, split):
    from pathlib import Path

    sample = Path(__file__).resolve().parents[1] / "data" / "sample"
    ref, omp = tmp_path / "ref", tmp_path / "omp"
    assert main(["detect", "--data", str(sample), "--out", str(ref)]) == 0
    assert main(["detect", "--backend", "openmp", "--threads", "4", "--split", split,
                 "--data", str(sample), "--out", str(omp)]) == 0
    assert main(["compare", str(ref), str(omp)]) == 0


def test_bench_openmp_scaling_on_sample(tmp_path):
    """The benchmark code (bench.py) runs end to end and writes the CSV and the chart."""
    from pathlib import Path

    from login_correlation import bench

    sample = Path(__file__).resolve().parents[1] / "data" / "sample"
    runs = bench.openmp_scaling(sample, threads=[1, 2], repeats=2)
    assert len(runs) == 2 * 2 * 2          # 2 splits x 2 thread counts x 2 repeats
    summary = bench.summarize_scaling(runs)
    assert (summary.loc[summary["threads"] == 1, "speedup"] == 1.0).all()
    bench.plot_scaling(summary, tmp_path / "chart.png")
    assert (tmp_path / "chart.png").stat().st_size > 0
