"""MPI backend (backends/mpi.py, milestone M4).

Two kinds of tests:
1. Without mpiexec: partition() + core.detect on every part + merge() simulates P ranks in
   one process. This checks the main idea - every account has exactly one owner, so the
   merged result equals the reference - for many values of P, quickly.
2. With mpiexec: the real program with real processes, compared with the reference.
"""

import json
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from conftest import random_events

from login_correlation import core
from login_correlation.__main__ import main
from login_correlation.backends import mpi, reference
from login_correlation.common import EventArrays, GeneratorParams, RuleParams
from login_correlation.generator import SCENARIO_TYPES, generate

PROJECT = Path(__file__).resolve().parents[1]
SAMPLE = PROJECT / "data" / "sample"
RANKS = [1, 2, 3, 4, 7, 16]


def same(a, b):
    return a.alarms.equals(b.alarms) and a.summary.equals(b.summary)


def simulate(events, params, size):
    """What `mpiexec -n size` does, but in one process: partition, detect per rank, merge."""
    p = mpi.partition(events, size)
    parts = []
    for r in range(size):
        res = core.detect(p.part(r), params)
        parts.append((res.alarms, res.summary))
    return mpi.merge(parts)


# ----------------------------------------------------------------- ownership


def test_owner_is_crc32_and_stable():
    assert mpi.owner_of("alice", 4) == zlib.crc32(b"alice") % 4
    assert all(0 <= mpi.owner_of(f"user{i}", 5) < 5 for i in range(1000))


def test_owner_is_the_same_in_another_process():
    """hash() would differ between processes (it is randomised); crc32 must not."""
    code = "from login_correlation.backends.mpi import owner_of; print(owner_of('user000123', 7))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONHASHSEED": "12345", "SYSTEMROOT": "C:\\Windows",
                              "PYTHONPATH": str(PROJECT / "src")})
    assert out.stdout.strip() == str(mpi.owner_of("user000123", 7))


def test_accounts_are_spread_evenly():
    names = [f"user{i:06d}" for i in range(10_000)]
    counts = np.bincount([mpi.owner_of(n, 4) for n in names], minlength=4)
    assert counts.min() > 0.9 * 2500 and counts.max() < 1.1 * 2500


# ----------------------------------------------------------------- partition


@pytest.mark.parametrize("size", RANKS)
def test_every_account_has_exactly_one_owner(size):
    ev = random_events(0)
    p = mpi.partition(ev, size)
    owned = [u for rank_users in p.users for u in rank_users]
    assert sorted(owned) == sorted(ev.users)          # every account once, none twice
    assert p.counts.sum() == len(ev)                  # no event lost or copied


@pytest.mark.parametrize("size", RANKS)
def test_rank_blocks_hold_only_owned_accounts_in_file_order(size):
    ev = random_events(1)
    p = mpi.partition(ev, size)
    for r in range(size):
        part = p.part(r)
        assert all(mpi.owner_of(u, size) == r for u in part.users)
        # inside a block the events keep their file order: per account, the timestamps
        # must appear in the same order as in the original data
        for local, name in enumerate(part.users[:5]):
            g = ev.users.index(name)
            assert np.array_equal(part.ts[part.user_id == local], ev.ts[ev.user_id == g])


# ----------------------------------------------------------------- simulated ranks = reference


@pytest.mark.parametrize("size", RANKS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_simulated_ranks_equal_reference(size, seed):
    ev = random_events(seed)
    for params in (RuleParams(60, 5), RuleParams(10, 2)):
        alarms, summary = simulate(ev, params, size)
        expected = reference.detect(ev, params)
        assert alarms.equals(expected.alarms) and summary.equals(expected.summary)
        assert alarms["alarm_id"].is_unique             # no alarm reported twice


def test_more_ranks_than_accounts():
    ev = EventArrays.from_rows([(t, "a", "ip", "FAIL") for t in range(5)], registry=["a", "b"])
    p = RuleParams(60, 3)
    alarms, summary = simulate(ev, p, 8)                 # at least 6 ranks get nothing
    expected = reference.detect(ev, p)
    assert alarms.equals(expected.alarms) and summary.equals(expected.summary)


def test_in_process_single_rank():
    """mpi.detect in this process = MPI with one rank."""
    pytest.importorskip("mpi4py")
    ev = random_events(3)
    p = RuleParams(60, 5)
    res = mpi.detect(ev, p)
    assert same(res, reference.detect(ev, p))
    assert res.info["ranks"] == 1 and res.info["events_per_rank"] == [len(ev)]
    assert {"partition", "scatter", "compute", "gather", "total"} <= set(res.timings)


# ----------------------------------------------------------------- real processes (mpiexec)


needs_mpiexec = pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec not found (install MS-MPI)")


def run_mpi(ranks, data, out, *extra):
    return subprocess.run(["mpiexec", "-n", str(ranks), sys.executable, "-m", "login_correlation",
                           "detect", "--backend", "mpi", "--data", str(data), "--out", str(out), *extra],
                          capture_output=True, text=True, timeout=300, cwd=PROJECT)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """A small generated dataset with every scenario type, and its reference result."""
    d = tmp_path_factory.mktemp("mpi_data")
    params = GeneratorParams(seed=21, n_users=800, n_events=40_000, duration_s=24 * 3600,
                             scenarios={t: 3 for t in SCENARIO_TYPES})
    generate(params, RuleParams(60, 5), d / "data")
    assert main(["detect", "--data", str(d / "data"), "--out", str(d / "ref")]) == 0
    return d


@needs_mpiexec
@pytest.mark.parametrize("ranks", [2, 3, 5, 8])
def test_mpiexec_on_generated_data(generated, tmp_path, ranks):
    out = tmp_path / f"mpi{ranks}"
    proc = run_mpi(ranks, generated / "data", out)
    assert proc.returncode == 0, proc.stderr
    assert main(["compare", str(generated / "ref"), str(out)]) == 0
    meta = json.loads((out / "timings.json").read_text())
    assert meta["ranks"] == ranks
    assert sum(meta["events_per_rank"]) == meta["n_events"]   # every event went to one rank
    assert len(meta["accounts_per_rank"]) == ranks


@needs_mpiexec
def test_mpiexec_with_other_rule_settings(generated, tmp_path):
    ref, out = tmp_path / "ref", tmp_path / "mpi"
    rule = ["--window", "10", "--threshold", "2"]
    assert main(["detect", "--data", str(generated / "data"), "--out", str(ref), *rule]) == 0
    proc = run_mpi(4, generated / "data", out, *rule)
    assert proc.returncode == 0, proc.stderr
    assert main(["compare", str(ref), str(out)]) == 0


@needs_mpiexec
def test_only_rank_0_writes_and_prints(tmp_path):
    proc = run_mpi(3, SAMPLE, tmp_path / "out")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("[mpi]") == 1                     # one summary line, from rank 0


# ----------------------------------------------------------------- benchmark code


def test_mpi_benchmark_summary_and_chart(tmp_path):
    """bench.summarize_mpi / plot_mpi on made-up run times (no mpiexec needed)."""
    import pandas as pd

    from login_correlation import bench

    runs = pd.DataFrame([
        {"ranks": r, "repeat": k, "n_events": 1000, "partition": 0.1, "scatter": 0.05,
         "compute": 0.8 / r, "gather": 0.05, "work": 0.2 + 0.8 / r,
         "local_min": 0.7 / r, "local_max": 0.8 / r}
        for r in (1, 2, 4) for k in range(3)
    ])
    s = bench.summarize_mpi(runs)
    assert s.loc[s["ranks"] == 1, "speedup"].iloc[0] == 1.0
    assert s.loc[s["ranks"] == 4, "speedup"].iloc[0] == pytest.approx(1.0 / 0.4)
    bench.plot_mpi(s, tmp_path / "mpi.png")
    assert (tmp_path / "mpi.png").stat().st_size > 0
