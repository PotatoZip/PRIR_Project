"""Synthetic login traffic with labelled attack scenarios (milestone M1).

Run:  python -m login_correlation generate --out data\\demo --events 1000000

Writes three files to <out_dir> (formats: docs/ARCHITECTURE.md section 5):
    events.csv        ts,user,src_ip,result   sorted by ts - the detector input
    users.csv         user                    all accounts, including ones with no events
    ground_truth.csv  scenario_id,type,user,start_ts,end_ts,n_events,expected_alarm

How the data is built:

1. Normal traffic. A "session" is one SUCCESS login. With probability `typo_rate` it is
   preceded by 1-2 FAILs a few seconds earlier (a mistyped password). Some users are much
   more active than others (activity ~ 1 / rank^0.8, capped at 20x the average user, so the
   busiest account is still a plausible person). Each user logs in from 1-3 home IPs
   in 10.0.0.0/8. Session times are uniform over [start_ts, start_ts + duration_s).

2. Scenarios (docs/ARCHITECTURE.md section 8.1) are inserted at random times. Each one gets
   its own victim account, and the victim's normal traffic is removed for 2*W seconds
   before and after the scenario (the "guard"). Without this, a random typo could turn an
   "edge, no alarm" scenario into an alarm, and the label would be wrong.
   Attackers use IPs from the documentation ranges 203.0.113.0/24 and 198.51.100.0/24.

3. All events are sorted by ts (stable, so equal timestamps keep their order) and written.

The same seed always gives byte-identical files.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from login_correlation.common import EMPTY_USER, FAIL, SUCCESS, GeneratorParams, RuleParams

SCENARIO_TYPES = ("brute_force", "slow_brute", "typo_burst", "edge_exact_window",
                  "edge_inside_window", "same_timestamp", "empty_username")
MAX_ACTIVITY = 20  # the busiest user logs in at most 20x as often as the average user
NEEDS_N_AT_LEAST_2 ={"slow_brute", "typo_burst", "edge_exact_window", "edge_inside_window"}


@dataclass
class Scenario:
    scenario_id: str
    type: str
    user: str                  # "" for empty_username (written as an empty CSV field)
    ts: list[int]              # timestamps of all scenario events
    results: list[str]
    ip: str
    expected_alarm: bool

    def failure_ts(self) -> list[int]:
        return [t for t, r in zip(self.ts, self.results) if r == FAIL]


def user_names(n: int, offset: int = 0) -> list[str]:
    return [f"user{i:06d}" for i in range(offset, offset + n)]


# ------------------------------------------------------------------------ normal traffic

def normal_traffic(p: GeneratorParams, rng: np.random.Generator):
    """Return (ts int64, user_idx int64, ip_idx int64, is_fail bool) for the normal traffic.

    ip_idx indexes the table built by home_ip_table (3 slots per user).
    """
    # expected events per session = 1 SUCCESS + typo_rate * 1.5 FAILs
    n_sessions = max(1, round(p.n_events / (1 + 1.5 * p.typo_rate)))

    weights = 1.0 / np.arange(1, p.n_users + 1) ** 0.8
    weights /= weights.sum()
    # Cap: no user is more than MAX_ACTIVITY x as active as the average user. Without the cap
    # the busiest account gets ~2 % of ALL logins (one every few seconds for a whole week),
    # its ordinary typos pile up into fake "attacks", and precision becomes meaningless.
    weights = np.minimum(weights, MAX_ACTIVITY / p.n_users)
    weights = rng.permutation(weights / weights.sum())         # activity not tied to the name
    s_user = rng.choice(p.n_users, size=n_sessions, p=weights)
    s_ts = p.start_ts + rng.integers(0, p.duration_s, n_sessions)

    n_home = rng.integers(1, 4, p.n_users)                     # 1-3 home IPs per user
    s_ip = s_user * 3 + (rng.random(n_sessions) * n_home[s_user]).astype(np.int64)

    # typos: 0, 1 or 2 FAILs before the SUCCESS, each 1-10 s earlier than the next event
    n_typos = np.where(rng.random(n_sessions) < p.typo_rate, rng.integers(1, 3, n_sessions), 0)
    t1 = n_typos >= 1
    t2 = n_typos >= 2
    gap1 = rng.integers(1, 11, n_sessions)
    gap2 = gap1 + rng.integers(1, 11, n_sessions)

    ts = np.concatenate([s_ts, s_ts[t1] - gap1[t1], s_ts[t2] - gap2[t2]])
    user = np.concatenate([s_user, s_user[t1], s_user[t2]])
    ip = np.concatenate([s_ip, s_ip[t1], s_ip[t2]])
    is_fail = np.concatenate([np.zeros(n_sessions, bool), np.ones(t1.sum(), bool), np.ones(t2.sum(), bool)])
    keep = ts >= p.start_ts                                     # typos before the start are dropped
    return ts[keep], user[keep], ip[keep], is_fail[keep]


def home_ip_table(n_users: int) -> np.ndarray:
    """3 possible home IPs per user: 10.<hi>.<lo>.<1..3>."""
    return np.array([f"10.{(u >> 8) & 255}.{u & 255}.{k + 1}"
                     for u in range(n_users) for k in range(3)], dtype=object)


# ------------------------------------------------------------------------ scenarios

def make_scenarios(p: GeneratorParams, rule: RuleParams, rng: np.random.Generator) -> list[Scenario]:
    W, N = rule.window_s, rule.threshold
    plan = [(t, int(p.scenarios.get(t, 0))) for t in SCENARIO_TYPES]
    if N < 2:   # with N = 1 every single failure is an alarm, so these cases cannot exist
        plan = [(t, c) for t, c in plan if t not in NEEDS_N_AT_LEAST_2]

    n_victims = sum(c for t, c in plan if t != "empty_username")
    if n_victims > p.n_users:
        raise ValueError(f"{n_victims} scenarios need {n_victims} victim accounts, "
                         f"but n_users is only {p.n_users}")
    margin = 2 * W + 5 * N * W                                 # room for the longest scenario + guard
    if p.duration_s <= 2 * margin:
        raise ValueError("duration_s is too short for the window and threshold")

    victims = iter(rng.choice(p.n_users, size=n_victims, replace=False).tolist())
    n_empty = dict(plan).get("empty_username", 0)
    scenarios: list[Scenario] = []

    for kind, count in plan:
        for k in range(count):
            if kind == "empty_username":
                # all of them share the account "<empty>", so they get separate time slots
                slot = (p.duration_s - 2 * margin) // max(n_empty, 1)
                t0 = p.start_ts + margin + k * slot + int(rng.integers(0, max(slot - margin, 1)))
                user = ""
            else:
                t0 = p.start_ts + margin + int(rng.integers(0, p.duration_s - 2 * margin))
                user = f"user{next(victims):06d}"
            ip = f"{rng.choice(['203.0.113', '198.51.100'])}.{int(rng.integers(1, 255))}"
            ts, results, expected = _scenario_events(kind, t0, W, N, rng)
            scenarios.append(Scenario(f"S{len(scenarios) + 1:04d}", kind, user, ts, results, ip, expected))
    return scenarios


def _scenario_events(kind: str, t0: int, W: int, N: int, rng: np.random.Generator):
    """Timestamps, results and expected_alarm of one scenario starting at t0."""
    def inside(n):              # n sorted timestamps strictly between t0 and t0 + W - 1
        return sorted((t0 + 1 + rng.integers(0, max(W - 2, 1), n)).tolist()) if n > 0 else []

    if kind == "brute_force":   # N..5N failures within W - 1 s; sometimes the attacker gets in
        n = int(rng.integers(N, 5 * N + 1))
        ts = [t0] + sorted((t0 + rng.integers(0, W, n - 1)).tolist())
        results = [FAIL] * n
        if rng.random() < 0.3:
            ts.append(ts[-1] + 1)
            results.append(SUCCESS)
        return ts, results, True
    if kind == "slow_brute":    # real attack, but too slow: gap g > (W-1)/(N-1) keeps every window < N
        gap = math.ceil(W / (N - 1)) + int(rng.integers(0, 5))
        n = int(rng.integers(2 * N, 3 * N + 1))
        return [t0 + i * gap for i in range(n)], [FAIL] * n, True
    if kind == "typo_burst":    # N-1 typos, then the right password
        ts = [t0 + 2 * i for i in range(N)]
        return ts, [FAIL] * (N - 1) + [SUCCESS], False
    if kind == "edge_exact_window":   # first and last failure exactly W apart -> no alarm
        return [t0] + inside(N - 2) + [t0 + W], [FAIL] * N, False
    if kind == "edge_inside_window":  # first and last failure W - 1 apart -> alarm
        return [t0] + inside(N - 2) + [t0 + W - 1], [FAIL] * N, True
    if kind == "same_timestamp":      # N..2N failures in the same second
        n = int(rng.integers(N, 2 * N + 1))
        return [t0] * n, [FAIL] * n, True
    if kind == "empty_username":      # brute force against an empty user name
        n = int(rng.integers(N, 2 * N + 1))
        return sorted((t0 + rng.integers(0, max(W // 2, 1), n)).tolist()), [FAIL] * n, True
    raise ValueError(f"unknown scenario type {kind!r}")


# ------------------------------------------------------------------------ main entry

def generate(params: GeneratorParams, rule: RuleParams, out_dir: str | Path) -> Path:
    """Create a dataset in out_dir and return the folder path."""
    started = time.perf_counter()
    rng = np.random.default_rng(params.seed)
    W = rule.window_s

    ts, user, ip, is_fail = normal_traffic(params, rng)
    scenarios = make_scenarios(params, rule, rng)

    # guard: remove the victim's normal traffic around each scenario
    keep = np.ones(ts.shape[0], dtype=bool)
    order = np.argsort(user, kind="stable")                    # rows grouped by user
    bounds = np.searchsorted(user[order], np.arange(params.n_users + 1))
    for s in scenarios:
        if s.user:
            u = int(s.user[4:])
            rows = order[bounds[u]:bounds[u + 1]]
            lo, hi = min(s.ts) - 2 * W, max(s.ts) + 2 * W
            keep[rows[(ts[rows] >= lo) & (ts[rows] <= hi)]] = False

    names = np.array(user_names(params.n_users), dtype=object)
    normal = pd.DataFrame({
        "ts": ts[keep],
        "user": names[user[keep]],
        "src_ip": home_ip_table(params.n_users)[ip[keep]],
        "result": np.where(is_fail[keep], FAIL, SUCCESS),
    })
    attack = pd.DataFrame(
        [(t, s.user, s.ip, r) for s in scenarios for t, r in zip(s.ts, s.results)],
        columns=["ts", "user", "src_ip", "result"],
    )
    events = pd.concat([normal, attack], ignore_index=True)
    events = events.sort_values("ts", kind="stable").reset_index(drop=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / "events.csv", index=False)

    registry = user_names(params.n_users) + user_names(params.n_empty_users, offset=params.n_users)
    pd.DataFrame({"user": registry}).to_csv(out / "users.csv", index=False)

    truth = pd.DataFrame([{
        "scenario_id": s.scenario_id,
        "type": s.type,
        "user": s.user or EMPTY_USER,          # the name the detector uses for an empty user field
        "start_ts": min(s.failure_ts()),
        "end_ts": max(s.failure_ts()),
        "n_events": len(s.ts),
        "expected_alarm": s.expected_alarm,
    } for s in scenarios], columns=["scenario_id", "type", "user", "start_ts", "end_ts",
                                     "n_events", "expected_alarm"])
    truth.to_csv(out / "ground_truth.csv", index=False)

    print(f"generated {len(events):,} events ({len(attack):,} in {len(scenarios)} scenarios), "
          f"{params.n_users:,} users + {params.n_empty_users} empty accounts "
          f"in {time.perf_counter() - started:.1f} s")
    return out
