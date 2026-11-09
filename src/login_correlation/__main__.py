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
    r.add_argument("--speedup", type=float, default=float("inf"))
    rule_args(r)

    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")

    b = sub.add_parser("bench")
    b.add_argument("--data", required=True)
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
            "threads": args.threads,
            "split": args.split,
            "device": args.device,
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


def run_compare(args) -> int:
    ok = True
    for name in ("alarms.csv", "summary.csv"):
        same = (Path(args.a) / name).read_bytes() == (Path(args.b) / name).read_bytes()
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
            from login_correlation.evaluate import evaluate

            metrics = evaluate(
                pd.read_csv(Path(args.results) / "alarms.csv"),
                pd.read_csv(Path(args.data) / "ground_truth.csv"),
            )
            (Path(args.results) / "metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )
            print(json.dumps(metrics, indent=2))
            return 0
        if args.cmd == "replay":
            from login_correlation.evaluate import replay

            df, registry = load_dataset(args.data)
            res = replay(encode(df, registry), rule_params(args, cfg), args.speedup)
            print(json.dumps(res["latency_s"], indent=2))
            return 0
        if args.cmd == "compare":
            return run_compare(args)
        if args.cmd == "bench":
            from login_correlation.bench import main as bench

            return bench(args.data, cfg.get("bench", {}))
    except NotImplementedError as exc:
        print(f"not implemented yet: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
