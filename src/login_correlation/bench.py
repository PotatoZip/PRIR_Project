"""Benchmarks and charts for the report.

Run:  python -m login_correlation bench --data data\\d10m

Done so far (M3-M5): OpenMP and MPI scaling, OpenCL phase breakdown.
Choose with --what all|openmp|mpi|opencl.

OpenMP
    threads from config [bench].threads, both splits, each setting repeated [bench].repeats
    times; the median is reported. The dataset is loaded and encoded once, and the Numba
    code is compiled before the first measurement, so only the detection itself is timed.
    -> results\\bench_openmp.csv           one row per run
    -> results\\plots\\speedup_openmp.png   time, speed-up and efficiency vs threads

MPI (M4)
    ranks from config [bench].ranks, each repeated [bench].mpi_repeats times (default 3).
    Each run is `mpiexec -n R python -m login_correlation detect --backend mpi`; the phase
    times are read from its timings.json.
    -> results\\bench_mpi.csv, results\\plots\\speedup_mpi.png

OpenCL (M5)
    load and encode measured once, the GPU pipeline [bench].repeats times; every phase timed
    -> results\\bench_opencl.csv, results\\plots\\opencl_phases.png

Still to do (M7, docs/ARCHITECTURE.md section 8.4):
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


MPI_PHASES = ("partition", "scatter", "compute", "gather")


def mpi_scaling(data_dir: str | Path, ranks: list[int], repeats: int, work_dir: Path) -> pd.DataFrame:
    """Run `mpiexec -n R python -m login_correlation detect --backend mpi` and read the phase
    times from timings.json. One row per run. Every run loads the CSV again on rank 0;
    loading is not part of the measured detection time."""
    import json
    import shutil
    import subprocess
    import sys

    if shutil.which("mpiexec") is None:
        raise RuntimeError("mpiexec not found - install MS-MPI and open a new terminal")
    rows = []
    for r in ranks:
        for rep in range(repeats):
            out = work_dir / f"mpi_n{r}"
            proc = subprocess.run(
                ["mpiexec", "-n", str(r), sys.executable, "-m", "login_correlation", "detect",
                 "--backend", "mpi", "--data", str(data_dir), "--out", str(out)],
                capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"mpiexec -n {r} failed:\n{proc.stderr}")
            meta = json.loads((out / "timings.json").read_text())
            t = meta["timings_s"]
            rows.append({"ranks": r, "repeat": rep, "n_events": meta["n_events"],
                         **{ph: t[ph] for ph in MPI_PHASES},
                         "work": t["prepare"] + t["compute"] + t["collect"],
                         "local_min": t["local_min"], "local_max": t["local_max"]})
        print(f"  mpi ranks={r:>2}: {pd.DataFrame(rows).query('ranks == @r')['work'].median():.3f} s")
    return pd.DataFrame(rows)


def summarize_mpi(runs: pd.DataFrame) -> pd.DataFrame:
    """Median per rank count + speed-up and efficiency, as for OpenMP."""
    med = runs.groupby("ranks", as_index=False)[[*MPI_PHASES, "work", "n_events",
                                                 "local_min", "local_max"]].median()
    base = med.loc[med["ranks"] == 1, "work"]
    t1 = float(base.iloc[0]) if len(base) else float(med["work"].iloc[0])
    med["speedup"] = t1 / med["work"]
    med["efficiency"] = med["speedup"] / med["ranks"]
    med["events_per_s"] = med["n_events"] / med["work"]
    return med


def plot_mpi(summary: pd.DataFrame, out_png: str | Path) -> None:
    """Left: where the time goes (stacked phases). Right: speed-up of the whole detection
    and of the local computation alone."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    x = range(len(summary))
    bottom = [0.0] * len(summary)
    colors = {"partition": "tab:gray", "scatter": "tab:orange", "compute": "tab:blue", "gather": "tab:green"}
    for ph in MPI_PHASES:
        ax[0].bar(x, summary[ph], bottom=bottom, color=colors[ph], label=ph)
        bottom = [b + v for b, v in zip(bottom, summary[ph])]
    ax[0].set_xticks(list(x), [str(r) for r in summary["ranks"]])
    ax[0].set(title="Where the time goes (median)", xlabel="MPI processes", ylabel="seconds")
    ax[0].legend()
    p_max = int(summary["ranks"].max())
    local_speedup = summary["compute"].iloc[0] / summary["compute"]
    ax[1].plot(summary["ranks"], summary["speedup"], "o-", label="whole detection")
    ax[1].plot(summary["ranks"], local_speedup, "s-", label="local computation only")
    ax[1].plot([1, p_max], [1, p_max], "k--", lw=1, label="ideal")
    ax[1].set(title="Speed-up S(p) = T(1) / T(p)", xlabel="MPI processes", ylabel="speed-up")
    for a in ax:
        a.grid(alpha=0.3)
    ax[1].legend()
    n = int(summary["n_events"].iloc[0])
    fig.suptitle(f"MPI (mpi4py + MS-MPI, one laptop) - {n:,} events - partition by account owner")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


GPU_PHASES = ("gpu_prepare", "h2d", "filter", "sort", "count", "stats", "d2h", "merge")


def opencl_phases(data_dir: str | Path, repeats: int, device: str | None = None) -> pd.DataFrame:
    """Time every phase of backends/opencl.py. Loading and encoding the CSV are measured once
    (they are the same for every run); the GPU pipeline is repeated `repeats` times.
    One row per run; load and encode are copied into every row."""
    import time

    from login_correlation.backends import opencl

    t = time.perf_counter()
    df, registry = load_dataset(data_dir)
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    events = encode(df, registry)
    encode_s = time.perf_counter() - t
    rows = []
    for r in range(repeats):
        res = opencl.detect(events, RuleParams(), device=device)
        rows.append({"repeat": r, "n_events": len(events), "load": load_s, "encode": encode_s,
                     **{ph: res.timings[ph] for ph in GPU_PHASES},
                     "gpu_total": res.timings["total"], "device": res.info["device"],
                     "failures": res.info["failures"]})
    print(f"  opencl on {rows[0]['device']}: median {pd.DataFrame(rows)['gpu_total'].median():.3f} s "
          f"(load {load_s:.1f} s, encode {encode_s:.1f} s)")
    return pd.DataFrame(rows)


def plot_opencl(runs: pd.DataFrame, out_png: str | Path) -> None:
    """Left: the whole way from CSV to alarms (shows how big encoding is).
    Right: the GPU pipeline alone, phase by phase."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    med = runs.median(numeric_only=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))

    whole = {"load CSV": med["load"], "encode (text -> numbers)": med["encode"],
             "GPU pipeline (all phases)": med["gpu_total"]}
    ax[0].barh(list(whole), list(whole.values()), color=["tab:gray", "tab:red", "tab:blue"])
    for k, (name, v) in enumerate(whole.items()):
        ax[0].text(v, k, f" {v:.2f} s", va="center")
    ax[0].set(title="From the CSV file to alarms", xlabel="seconds")
    ax[0].invert_yaxis()

    labels = {"gpu_prepare": "prepare (CPU)", "h2d": "copy to GPU", "filter": "filter (GPU)",
              "sort": "radix sort (GPU)", "count": "window count (GPU)",
              "stats": "per-user stats (GPU)", "d2h": "copy back", "merge": "alarms + tables (CPU)"}
    vals = [med[ph] * 1000 for ph in GPU_PHASES]
    ax[1].barh([labels[ph] for ph in GPU_PHASES], vals,
               color=["tab:orange" if ph in ("gpu_prepare", "merge") else
                      "tab:purple" if ph in ("h2d", "d2h") else "tab:blue" for ph in GPU_PHASES])
    for k, v in enumerate(vals):
        ax[1].text(v, k, f" {v:.1f} ms", va="center")
    ax[1].set(title="GPU pipeline, phase by phase (median)", xlabel="milliseconds")
    ax[1].invert_yaxis()
    for a in ax:
        a.grid(alpha=0.3, axis="x")
    n = int(med["n_events"])
    fig.suptitle(f"OpenCL (PyOpenCL) on {runs['device'].iloc[0]} - {n:,} events, "
                 f"{int(med['failures']):,} failures")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(data_dir: str, settings: dict, out_dir: str = "results", what: str = "all") -> int:
    """settings = the [bench] section of config/default.toml. what = all | openmp | mpi | opencl."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    repeats = int(settings.get("repeats", 5))

    if what in ("all", "openmp"):
        threads = [t for t in settings.get("threads", DEFAULT_THREADS) if t <= (os.cpu_count() or 1)]
        print(f"OpenMP scaling on {data_dir} (repeats={repeats})")
        runs = openmp_scaling(data_dir, threads, repeats)
        runs.to_csv(out / "bench_openmp.csv", index=False)
        summary = summarize_scaling(runs)
        plot_scaling(summary, out / "plots" / "speedup_openmp.png")
        cols = ["split", "threads", *PHASES, "work", "speedup", "efficiency", "events_per_s"]
        print(summary[cols].round(3).to_string(index=False))
        print(f"runs: {out / 'bench_openmp.csv'}   chart: {out / 'plots' / 'speedup_openmp.png'}\n")

    if what in ("all", "mpi"):
        ranks = [r for r in settings.get("ranks", [1, 2, 4, 6, 8]) if r <= (os.cpu_count() or 1)]
        mpi_repeats = int(settings.get("mpi_repeats", 3))
        print(f"MPI scaling on {data_dir} (repeats={mpi_repeats}; every run reloads the CSV, "
              "so this takes a few minutes)")
        runs = mpi_scaling(data_dir, ranks, mpi_repeats, out / "bench_mpi_runs")
        runs.to_csv(out / "bench_mpi.csv", index=False)
        summary = summarize_mpi(runs)
        plot_mpi(summary, out / "plots" / "speedup_mpi.png")
        cols = ["ranks", *MPI_PHASES, "work", "speedup", "efficiency", "local_min", "local_max"]
        print(summary[cols].round(3).to_string(index=False))
        print(f"runs: {out / 'bench_mpi.csv'}   chart: {out / 'plots' / 'speedup_mpi.png'}")

    if what in ("all", "opencl"):
        print(f"OpenCL phases on {data_dir} (repeats={repeats})")
        runs = opencl_phases(data_dir, repeats, settings.get("device"))
        runs.to_csv(out / "bench_opencl.csv", index=False)
        plot_opencl(runs, out / "plots" / "opencl_phases.png")
        med = runs.median(numeric_only=True)
        for ph in ("load", "encode", *GPU_PHASES, "gpu_total"):
            print(f"  {ph:<12} {med[ph] * 1000:10.1f} ms")
        print(f"runs: {out / 'bench_opencl.csv'}   chart: {out / 'plots' / 'opencl_phases.png'}")

    print("(the data-size experiment follows in milestone M7)")
    return 0
