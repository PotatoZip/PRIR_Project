"""Shared building blocks used by every part of the project:

* load_config                   - reads config/default.toml
* RuleParams / GeneratorParams  - parameters (docs/ARCHITECTURE.md section 2)
* EventArrays                   - the events as integer NumPy arrays (section 5)
* DetectionResult               - what every backend returns
* PhaseTimer                    - measures the phases load/encode/prepare/compute/merge/collect
* load_dataset / encode / write_result - CSV input and output
* get_backend                   - picks the backend module by name
"""

from __future__ import annotations

import importlib
import json
import time
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- parameters

PROJECT_DIR = Path(__file__).resolve().parents[2]          # src/login_correlation/common.py -> project
DEFAULT_CONFIG = PROJECT_DIR / "config" / "default.toml"


def load_config(path: str | Path | None = None) -> dict:
    """Read the TOML config (default: config/default.toml). Missing file -> {} (built-in defaults)."""
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        return {}
    with p.open("rb") as fh:
        return tomllib.load(fh)


@dataclass(frozen=True)
class RuleParams:
    """The one rule: at least `threshold` failures of one user inside (t - window_s, t]."""

    window_s: int = 60    # W
    threshold: int = 5    # N

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError("window_s must be > 0")
        if self.threshold < 1:
            raise ValueError("threshold must be >= 1")


@dataclass(frozen=True)
class GeneratorParams:
    """Synthetic data settings (docs/ARCHITECTURE.md section 8)."""

    seed: int = 42
    n_users: int = 10_000
    n_events: int = 1_000_000
    start_ts: int = 1_767_225_600      # 2026-01-01T00:00:00Z
    duration_s: int = 7 * 24 * 3600    # one week
    typo_rate: float = 0.03            # share of normal logins preceded by 1-2 FAILs
    n_empty_users: int = 50            # accounts in users.csv with no events
    scenarios: dict[str, int] = field(default_factory=lambda: {
        "brute_force": 40, "slow_brute": 10, "typo_burst": 30,
        "edge_exact_window": 10, "edge_inside_window": 10, "same_timestamp": 10,
        "empty_username": 3,
    })


# ----------------------------------------------------------------------------- data model

FAIL, SUCCESS = "FAIL", "SUCCESS"
EMPTY_USER = "<empty>"   # account name used for events whose user field is empty
EVENT_COLUMNS = ["ts", "user", "src_ip", "result"]
ALARM_COLUMNS = ["alarm_id", "user", "start_ts", "end_ts", "trigger_ts", "n_failures"]
SUMMARY_COLUMNS = ["user", "n_events", "n_fail", "n_success",
                   "first_ts", "last_ts", "max_window_count", "n_alarms"]


@dataclass
class EventArrays:
    """Events as columns of numbers. Row index == line number in events.csv (`seq`).

    users[user_id] is the account name; `users` also contains accounts that have
    no events (the "empty accounts" from users.csv).
    """

    ts: np.ndarray        # int64, epoch seconds
    user_id: np.ndarray   # int32
    ip_id: np.ndarray     # int32
    is_fail: np.ndarray   # uint8, 1 = FAIL
    users: list[str]
    ips: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    @classmethod
    def from_rows(cls, rows: Iterable[tuple], registry: Iterable[str] = ()) -> "EventArrays":
        """Helper for tests: build from (ts, user, src_ip, result) tuples."""
        return encode(pd.DataFrame(list(rows), columns=EVENT_COLUMNS), list(registry))


@dataclass
class DetectionResult:
    alarms: pd.DataFrame       # ALARM_COLUMNS, sorted by (user, start_ts)
    summary: pd.DataFrame      # SUMMARY_COLUMNS, sorted by user
    timings: dict[str, float] = field(default_factory=dict)


def make_alarm_id(user: str, start_ts: int) -> str:
    return f"{user}:{start_ts}"


ALARM_DTYPES = {"alarm_id": object, "user": object, "start_ts": "int64",
                "end_ts": "int64", "trigger_ts": "int64", "n_failures": "int64"}
SUMMARY_DTYPES = {"user": object, "n_events": "int64", "n_fail": "int64", "n_success": "int64",
                  "first_ts": "Int64", "last_ts": "Int64", "max_window_count": "int64",
                  "n_alarms": "int64"}


def normalize_alarms(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed columns, types and row order, so the outputs of all backends are identical
    (as DataFrames and as CSV files). Every backend must pass its alarms through this."""
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in ALARM_DTYPES.items()})
    out = df[ALARM_COLUMNS].astype(ALARM_DTYPES)
    return out.sort_values(["user", "start_ts"], kind="stable").reset_index(drop=True)


def normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Same as normalize_alarms, for the per-account summary (sorted by user)."""
    out = df[SUMMARY_COLUMNS].astype(SUMMARY_DTYPES)
    return out.sort_values("user", kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------- timing

class PhaseTimer:
    """with timer.phase("compute"): ...   ->   timer.as_dict() == {"compute": seconds}"""

    def __init__(self) -> None:
        self._acc: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def add(self, name: str, seconds: float) -> None:
        """For times measured elsewhere (OpenCL profiling events, MPI.Wtime)."""
        self._acc[name] = self._acc.get(name, 0.0) + seconds

    def as_dict(self) -> dict[str, float]:
        return dict(self._acc)


# ----------------------------------------------------------------------------- input / output

def load_dataset(data_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Read <data_dir>/events.csv and the optional account registry <data_dir>/users.csv."""
    data_dir = Path(data_dir)
    dtypes = {"ts": "int64", "user": "string", "src_ip": "string", "result": "string"}
    try:  # the pyarrow engine is much faster for big files, if it is installed
        df = pd.read_csv(data_dir / "events.csv", dtype=dtypes, engine="pyarrow")
    except (ImportError, ValueError):
        df = pd.read_csv(data_dir / "events.csv", dtype=dtypes)
    users_csv = data_dir / "users.csv"
    registry = pd.read_csv(users_csv, dtype="string")["user"].tolist() if users_csv.exists() else []
    return df, registry


def encode(df: pd.DataFrame, registry: list[str] | None = None) -> EventArrays:
    """Text columns -> integer arrays (dictionary encoding).

    User ids: registry accounts first (in registry order), then accounts that only appear
    in the events (in order of first appearance). Same input -> same ids.
    """
    # 1. factorize: every row gets the index of its name in `uniques` (first appearance
    #    order). This is the only step that touches all rows, and it runs in C. A missing
    #    value (empty CSV field) gets code -1.
    codes, uniques = pd.factorize(df["user"], use_na_sentinel=True)

    # 2. Build the final name list (registry first) and a small table that maps a
    #    factorize code to a final id. This loop runs once per distinct name, not per row.
    #    An empty user field is a real case (a login attempt without a name). It is kept as
    #    its own account called EMPTY_USER instead of being dropped or crashing.
    users = list(registry or [])
    index = {name: i for i, name in enumerate(users)}

    def id_of(name: str) -> int:
        if name not in index:
            index[name] = len(users)
            users.append(name)
        return index[name]

    code_to_id = np.array([id_of(str(u).strip() or EMPTY_USER) for u in uniques] + [0],
                          dtype=np.int32)
    if (codes < 0).any():
        code_to_id[-1] = id_of(EMPTY_USER)          # code -1 reads the last table entry
    user_id = code_to_id[codes]

    ip_codes, ip_names = pd.factorize(df["src_ip"], use_na_sentinel=False)
    return EventArrays(
        ts=df["ts"].to_numpy(dtype=np.int64),
        user_id=user_id,
        ip_id=ip_codes.astype(np.int32),
        is_fail=(df["result"] == FAIL).to_numpy(dtype=bool, na_value=False).astype(np.uint8),
        users=users,
        ips=[str(ip) for ip in ip_names],
    )


def write_result(result: DetectionResult, out_dir: str | Path, meta: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.alarms.to_csv(out / "alarms.csv", index=False)
    result.summary.to_csv(out / "summary.csv", index=False)
    (out / "timings.json").write_text(
        json.dumps({"timings_s": result.timings, **(meta or {})}, indent=2), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- backends

BACKENDS = {
    "reference": "login_correlation.backends.reference",  # plain Python, the correct answer - done
    "openmp":    "login_correlation.backends.openmp",        # Numba prange on OpenMP            - TODO M3
    "mpi":       "login_correlation.backends.mpi",        # mpi4py + MS-MPI                   - TODO M4
    "opencl":    "login_correlation.backends.opencl",        # PyOpenCL on the Iris Xe GPU       - TODO M5
}


def get_backend(name: str) -> Callable[..., DetectionResult]:
    """Import the backend only when it is used, so a missing mpi4py/pyopencl breaks only that backend."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return importlib.import_module(BACKENDS[name]).detect
