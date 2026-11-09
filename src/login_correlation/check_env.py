"""Checks that every tool the project needs works.   Run: python -m login_correlation check

Prints one line per component; returns 0 when everything is OK.
How to fix a problem: README.md, section "Setup".
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"


class Report:
    def __init__(self) -> None:
        self.problems = 0

    def __call__(self, status: str, name: str, detail: str) -> None:
        self.problems += status == FAIL
        print(f"{status} {name:<20} {detail}")


def main() -> int:
    r = Report()
    print("login_correlation environment check\n" + "-" * 60)

    r(OK if sys.version_info >= (3, 11) else FAIL, "Python", platform.python_version())
    in_venv = sys.prefix != sys.base_prefix
    r(OK if in_venv else WARN, "virtualenv", "active" if in_venv else "not active (use .venv)")
    r(OK, "CPU", f"{platform.processor()} | logical cores: {os.cpu_count()}")
    for mod in ("numpy", "pandas", "numba", "matplotlib", "pytest"):
        try:
            r(OK, mod, __import__(mod).__version__)
        except Exception as exc:  # noqa: BLE001
            r(FAIL, mod, str(exc))

    # OpenMP through Numba
    os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
    try:
        import numba
        from numba import njit, prange

        @njit(parallel=True)
        def _sum(n):
            s = 0
            for i in prange(n):
                s += i
            return s

        assert _sum(1000) == 499500
        layer = numba.threading_layer()
        r(OK if layer == "omp" else WARN, "OpenMP (Numba)",
          f"layer={layer}, threads={numba.config.NUMBA_NUM_THREADS}")
    except Exception as exc:  # noqa: BLE001
        r(FAIL, "OpenMP (Numba)", str(exc))

    # MPI
    exe = shutil.which("mpiexec")
    r(OK if exe else FAIL, "mpiexec", exe or "not found - install MS-MPI and open a NEW terminal")
    try:
        from mpi4py import MPI
        r(OK, "mpi4py", MPI.Get_library_version().splitlines()[0].strip())
        if exe:
            out = subprocess.run([exe, "-n", "2", sys.executable, "-c",
                                  "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"],
                                 capture_output=True, text=True, timeout=60)
            ok = out.stdout.split() == ["2", "2"]
            r(OK if ok else FAIL, "mpiexec -n 2", "2 processes started" if ok else out.stderr[:200])
    except Exception as exc:  # noqa: BLE001
        r(FAIL, "mpi4py", str(exc).splitlines()[0])

    # OpenCL
    try:
        import numpy as np
        import pyopencl as cl
        import pyopencl.array as cla

        for pi, p in enumerate(cl.get_platforms()):
            for di, d in enumerate(p.get_devices()):
                r(OK, f"OpenCL {pi}:{di}", f"{d.name.strip()} [{cl.device_type.to_string(d.type)}] "
                                            f"on {p.name.strip()}")
        ctx = cl.create_some_context(interactive=False)
        q = cl.CommandQueue(ctx)
        a = cla.to_device(q, np.arange(1024, dtype=np.int32))
        ok = int((a * 2).get().sum()) == 2 * sum(range(1024))
        r(OK if ok else FAIL, "OpenCL kernel run", ctx.devices[0].name.strip())
    except Exception as exc:  # noqa: BLE001
        r(FAIL, "OpenCL", str(exc).splitlines()[0])

    print("-" * 60)
    print("All good." if r.problems == 0 else f"{r.problems} problem(s) - see README.md, Setup")
    return 0 if r.problems == 0 else 1
