"""Command line:  python -m login_correlation <command>

check      check that OpenMP, MPI and OpenCL work on this machine
generate   create a synthetic dataset with labelled attacks
detect     run one backend: reference | openmp | mpi | opencl
evaluate   precision / recall against ground_truth.csv
replay     stream replay, time-to-alarm
compare    check that two result folders are identical
bench      run the benchmarks and draw the charts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from login_correlation.common import (
    BACKENDS,
    GeneratorParams,
    PhaseTimer,
    RuleParams,
    encode,
    get_backend,
    load_config,
    load_dataset,
    write_result,
)


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="python -m login_correlation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", help="settings file (default: config/default.toml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def rule_args(sp):  # default None = take the value from the config file
        sp.add_argument(
            "--window",
            type=int,
            help="W, window length in seconds (config: rule.window_s)",
        )
        sp.add_argument(
            "--threshold",
            type=int,
            help="N, failures that raise an alarm (config: rule.threshold)",
        )

    sub.add_parser("check")

    g = sub.add_parser("generate")
    g.add_argument("--out", required=True, help="output folder, e.g. data\\demo")
    g.add_argument(
        "--events", type=int, help="number of events (config: generator.n_events)"
    )
    g.add_argument(
        "--users", type=int, help="number of accounts (config: generator.n_users)"
    )
    g.add_argument("--seed", type=int, help="random seed (config: generator.seed)")
    rule_args(g)

    d = sub.add_parser("detect")
    d.add_argument("--backend", choices=sorted(BACKENDS), default="reference")
    d.add_argument("--data", required=True, help="folder with events.csv (+ users.csv)")
    d.add_argument(
        "--out", required=True, help="folder for alarms.csv, summary.csv, timings.json"
    )
    d.add_argument("--threads", type=int, help="openmp: number of threads")
    d.add_argument(
        "--split",
        choices=["users", "time"],
        default="users",
        help="openmp: split by accounts (default) or by time chunks with a halo",
    )
    d.add_argument("--device", help="opencl: platform:device, e.g. 0:0")
    rule_args(d)

    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True)
    e.add_argument("--results", required=True)

    r = sub.add_parser("replay")
    r.add_argument("--data", required=True)
    r.add_argument("--speedup", type=float, default=float("inf"),
                   help="seconds of log time per second of real time (default inf = as fast as possible)")
    r.add_argument("--out", help="folder for alarms.csv, per_alarm.csv and replay.json (optional)")
    r.add_argument("--no-check", action="store_true",
                   help="do not compare the replayed alarms with batch detection")
    rule_args(r)

    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")

    b = sub.add_parser("bench")
    b.add_argument("--data", required=True)
    b.add_argument("--what", choices=["all", "openmp", "mpi", "opencl", "stream"], default="all",
                   help="which benchmark to run (default: all)")
    return p.parse_args(argv)


def rule_params(args, cfg) -> RuleParams:
    """Command-line value if given, otherwise the config file, otherwise the built-in default."""
    rule = cfg.get("rule", {})
    return RuleParams(
        window_s=args.window if args.window is not None else rule.get("window_s", 60),
        threshold=(
            args.threshold if args.threshold is not None else rule.get("threshold", 5)
        ),
    )


def generator_params(args, cfg) -> GeneratorParams:
    gen = dict(cfg.get("generator", {}))
    for key, value in (
        ("n_events", args.events),
        ("n_users", args.users),
        ("seed", args.seed),
    ):
        if value is not None:
            gen[key] = value
    return GeneratorParams(**gen)


def run_detect(args, cfg) -> int:
    params = rule_params(args, cfg)
    detect = get_backend(args.backend)

    is_worker = False
    if args.backend == "mpi":  # with MPI only rank 0 reads the input
        from mpi4py import MPI

        is_worker = MPI.COMM_WORLD.Get_rank() != 0

    timer, events = PhaseTimer(), None
    if not is_worker:
        with timer.phase("load"):
            df, registry = load_dataset(args.data)
        with timer.phase("encode"):
            events = encode(df, registry)

    result = detect(
        events, params, threads=args.threads, split=args.split, device=args.device
    )
    if result is None:  # MPI ranks other than 0
        return 0

    result.timings = {**timer.as_dict(), **result.timings}
    out = write_result(
        result,
        args.out,
        {
            "backend": args.backend,
            "window_s": params.window_s,
            "threshold": params.threshold,
            "n_events": len(events),
            # options that only one backend uses are recorded only for that backend
            "threads": args.threads if args.backend in ("openmp", "mpi") else None,
            "split": args.split if args.backend == "openmp" else None,
            "device": args.device if args.backend == "opencl" else None,
            **result.info,
        },
    )
    work = sum(
        result.timings.get(k, 0.0) for k in ("prepare", "compute", "merge", "collect")
    )
    print(
        f"[{args.backend}] {len(events):,} events -> {len(result.alarms):,} alarms | "
        f"{len(events) / work if work else float('nan'):,.0f} events/s (without load/encode) | {out}"
    )
    return 0


def run_replay(args, cfg) -> int:
    """Replay the dataset as a live stream (stream.py) and report time-to-alarm."""
    from login_correlation import core
    from login_correlation.stream import format_report, replay

    params = rule_params(args, cfg)
    df, registry = load_dataset(args.data)
    events = encode(df, registry)
    res = replay(events, params, args.speedup)
    identical = None
    if not args.no_check:  # the fast batch core, which is tested to equal the reference
        identical = bool(res["alarms"].equals(core.detect(events, params).alarms))
    print(format_report(res["stats"], identical))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        res["alarms"].to_csv(out / "alarms.csv", index=False)
        res["per_alarm"].to_csv(out / "per_alarm.csv", index=False)
        (out / "replay.json").write_text(json.dumps(
            {**res["stats"], "window_s": params.window_s, "threshold": params.threshold,
             "same_as_batch": identical}, indent=2), encoding="utf-8")
        print(f"(results saved to {out})")
    return 0 if identical in (None, True) else 1


def run_compare(args) -> int:
    """Byte-for-byte comparison of two result folders. alarms.csv must be in both; summary.csv
    is skipped when one folder has none (e.g. `replay` writes alarms only)."""
    ok = True
    for name in ("alarms.csv", "summary.csv"):
        a, b = Path(args.a) / name, Path(args.b) / name
        if name == "summary.csv" and not (a.exists() and b.exists()):
            print(f"{name}: skipped (not in both folders)")
            continue
        same = a.read_bytes() == b.read_bytes()
        ok &= same
        print(f"{name}: {'IDENTICAL' if same else 'DIFFERENT'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    try:
        if args.cmd == "check":
            from login_correlation.check_env import main as check

            return check()
        if args.cmd == "generate":
            from login_correlation.generator import generate

            out = generate(
                generator_params(args, cfg), rule_params(args, cfg), args.out
            )
            print(f"dataset written to {out}")
            return 0
        if args.cmd == "detect":
            return run_detect(args, cfg)
        if args.cmd == "evaluate":
            from login_correlation.evaluate import evaluate, format_report

            text = {"user": str}  # account names are text, even if they look like numbers
            metrics = evaluate(
                pd.read_csv(Path(args.results) / "alarms.csv", dtype=text),
                pd.read_csv(Path(args.data) / "ground_truth.csv", dtype=text),
            )
            (Path(args.results) / "metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )
            print(format_report(metrics))
            print(f"\n(full metrics saved to {Path(args.results) / 'metrics.json'})")
            return 0
        if args.cmd == "replay":
            return run_replay(args, cfg)
        if args.cmd == "compare":
            return run_compare(args)
        if args.cmd == "bench":
            from login_correlation.bench import main as bench

            return bench(args.data, cfg.get("bench", {}), what=args.what)
    except NotImplementedError as exc:
        print(f"not implemented yet: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
