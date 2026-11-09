"""Run: python -m login_correlation bench --data data\\d10m

Experiments (docs/ARCHITECTURE.md section 8; numbers in config/default.toml [bench]), each repeated 5 times, median reported:
    openmp  threads 1, 2, 4, 6, 8, 10, 12        (in-process)
    mpi     ranks 1, 2, 4, 6, 8                  (subprocess: mpiexec -n R python -m login_correlation detect ...,
                                                  then read results\\...\\timings.json)
    opencl  phase breakdown encode / h2d / filter / sort / count / d2h / merge
    all     sizes 0.1M, 1M, 5M, 10M  -> events/s
    quality W in {10, 60, 300}, N in {3, 5, 10} -> precision / recall

One CSV row per run -> results\\bench.csv; charts (matplotlib PNG) -> results\\plots\\:
    speedup_openmp.png, speedup_mpi.png, throughput.png, opencl_phases.png,
    quality_params.png, stream_latency.png
"""

from __future__ import annotations


def main(data_dir: str, settings: dict, out_csv: str = "results/bench.csv") -> int:
    """settings = the [bench] section of config/default.toml (repeats, sizes, threads, ranks). TODO(M7)."""
    raise NotImplementedError("benchmarks - milestone M7")
