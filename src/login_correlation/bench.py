"""Benchmarks and charts for the report.

Run:  python -m login_correlation bench --data data\\d10m

Done so far (M3): OpenMP strong scaling
    threads from config [bench].threads, both splits, each setting repeated [bench].repeats
    times; the median is reported. The dataset is loaded and encoded once, and the Numba
    code is compiled before the first measurement, so only the detection itself is timed.
    -> results\\bench_openmp.csv           one row per run
    -> results\\plots\\speedup_openmp.png   time, speed-up and efficiency vs threads

Still to do (M7, docs/ARCHITECTURE.md section 8.4):
    mpi     ranks 1, 2, 4, 6, 8  (subprocess: mpiexec -n R python -m login_correlation detect ...,
                                   then read results\\...\\timings.json)
    opencl  phase breakdown encode / h2d / filter / sort / count / d2h / merge
    all     sizes 0.1M, 1M, 5M, 10M  -> events/s
    quality W in {10, 60, 300}, N in {3, 5, 10} -> precision / recall
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from login_correlation.common import RuleParams, encode, load_dataset

DEFAULT_THREADS = [1, 2, 4, 6, 8, 10, 12]
PHASES = ("prepare", "compute", "merge")


def openmp_scaling(data_dir: str | Path, threads: list[int], repeats: int,
                   rule: RuleParams = RuleParams()) -> pd.DataFrame:
    """Time backends/openmp.py for every (split, threads) setting. One row per run."""
    from login_correlation.backends import openmp

    df, registry = load_dataset(data_dir)
    events = encode(df, registry)
    openmp.warm_up()
    rows = []
    for split in ("users", "time"):
        for t in threads:
            for r in range(repeats):
                res = openmp.detect(events, rule, threads=t, split=split)
                tm = res.timings
                rows.append({"split": split, "threads": t, "repeat": r, "n_events": len(events),
                             **{p: tm.get(p, 0.0) for p in PHASES},
                             "work": sum(tm.get(p, 0.0) for p in PHASES)})
            print(f"  openmp split={split:<5} threads={t:>2}: "
                  f"{pd.DataFrame(rows).query('split == @split and threads == @t')['work'].median():.3f} s")
    return pd.DataFrame(rows)


def summarize_scaling(runs: pd.DataFrame) -> pd.DataFrame:
    """Median per (split, threads) + speed-up S(p) = T(1)/T(p) and efficiency E(p) = S(p)/p."""
    med = runs.groupby(["split", "threads"], as_index=False)[[*PHASES, "work", "n_events"]].median()
    base = med[med["threads"] == 1].set_index("split")["work"]
    med["speedup"] = med["split"].map(base) / med["work"]
    med["efficiency"] = med["speedup"] / med["threads"]
    med["events_per_s"] = med["n_events"] / med["work"]
    return med


def plot_scaling(summary: pd.DataFrame, out_png: str | Path) -> None:
    """Three panels: time per phase, speed-up (with the ideal line), efficiency."""
    import matplotlib

    matplotlib.use("Agg")  # draw to a file, no window needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    styles = {"users": ("tab:blue", "o", "split by accounts"),
              "time": ("tab:orange", "s", "split by time + halo")}
    for split, (color, marker, label) in styles.items():
        s = summary[summary["split"] == split]
        ax[0].plot(s["threads"], s["work"], marker=marker, color=color, label=label)
        ax[1].plot(s["threads"], s["speedup"], marker=marker, color=color, label=label)
        ax[2].plot(s["threads"], s["efficiency"], marker=marker, color=color, label=label)
    p_max = summary["threads"].max()
    ax[1].plot([1, p_max], [1, p_max], "k--", lw=1, label="ideal")
    ax[0].set(title="Detection time (prepare + compute + merge)", xlabel="threads", ylabel="seconds")
    ax[1].set(title="Speed-up S(p) = T(1) / T(p)", xlabel="threads", ylabel="speed-up")
    ax[2].set(title="Efficiency E(p) = S(p) / p", xlabel="threads", ylabel="efficiency", ylim=(0, 1.1))
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    n = int(summary["n_events"].iloc[0])
    fig.suptitle(f"OpenMP (Numba prange) - {n:,} events - Intel i5-1235U (2 P-cores + 8 E-cores, 12 threads)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(data_dir: str, settings: dict, out_dir: str = "results") -> int:
    """settings = the [bench] section of config/default.toml (repeats, threads, ...)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    threads = [t for t in settings.get("threads", DEFAULT_THREADS) if t <= (os.cpu_count() or 1)]
    repeats = int(settings.get("repeats", 5))

    print(f"OpenMP scaling on {data_dir} (repeats={repeats})")
    runs = openmp_scaling(data_dir, threads, repeats)
    runs.to_csv(out / "bench_openmp.csv", index=False)
    summary = summarize_scaling(runs)
    plot_scaling(summary, out / "plots" / "speedup_openmp.png")
    cols = ["split", "threads", *PHASES, "work", "speedup", "efficiency", "events_per_s"]
    print(summary[cols].round(3).to_string(index=False))
    print(f"\nruns: {out / 'bench_openmp.csv'}   chart: {out / 'plots' / 'speedup_openmp.png'}")
    print("(MPI, OpenCL and data-size experiments follow in milestone M7)")
    return 0
