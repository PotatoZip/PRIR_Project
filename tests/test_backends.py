"""Every parallel backend must give exactly the same result as the reference backend.

Backends that are not written yet are skipped, so these tests turn green
milestone by milestone (M3 openmp, M4 mpi, M5 opencl).
The OpenMP time split is tested with chunk borders that cut through an attack, which checks
correctness at the borders of data chunks.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import random_events

from login_correlation.__main__ import main
from login_correlation.backends.reference import detect
from login_correlation.common import RuleParams, get_backend

PROJECT = Path(__file__).resolve().parents[1]
SAMPLE = PROJECT / "data" / "sample"


BACKEND_VARIANTS = [
    ("openmp", {"split": "users", "threads": 1}),
    ("openmp", {"split": "users", "threads": 4}),
    ("openmp", {"split": "time", "threads": 2}),    # chunk border near the attack
    ("openmp", {"split": "time", "threads": 7}),
    ("opencl", {}),
]


@pytest.mark.parametrize("backend, options", BACKEND_VARIANTS)
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("params", [RuleParams(60, 5), RuleParams(10, 2), RuleParams(300, 10)])
def test_backend_matches_reference(backend, options, seed, params):
    events = random_events(seed)
    expected = detect(events, params)
    try:
        got = get_backend(backend)(events, params, **options)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    assert got.alarms.equals(expected.alarms)
    assert got.summary.equals(expected.summary)


@pytest.mark.parametrize("ranks", [1, 2, 3, 4])
def test_mpi_matches_reference(tmp_path, ranks):
    """MPI must be started with mpiexec, so this test runs it as a separate program."""
    if not shutil.which("mpiexec"):
        pytest.skip("mpiexec not found (install MS-MPI)")
    ref, out = tmp_path / "ref", tmp_path / f"mpi{ranks}"
    main(["detect", "--data", str(SAMPLE), "--out", str(ref)])
    proc = subprocess.run(["mpiexec", "-n", str(ranks), sys.executable, "-m", "login_correlation",
                           "detect", "--backend", "mpi", "--data", str(SAMPLE), "--out", str(out)],
                          capture_output=True, text=True, timeout=120, cwd=PROJECT)
    if "not implemented yet" in proc.stderr:
        pytest.skip("mpi backend - milestone M4")
    assert proc.returncode == 0, proc.stderr
    assert main(["compare", str(ref), str(out)]) == 0
