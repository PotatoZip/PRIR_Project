"""Measuring the results: detection quality (milestone M2) and stream replay (TODO M6).

Quality (docs/ARCHITECTURE.md section 8.2):
    an alarm MATCHES a scenario if it has the same user and the time ranges overlap:
        alarm.start_ts <= scenario.end_ts  and  alarm.end_ts >= scenario.start_ts
    precision = alarms that match a scenario with expected_alarm=True / all alarms
    recall    = scenarios with expected_alarm=True found by >= 1 alarm / all such scenarios
    f1        = 2 * precision * recall / (precision + recall)
    delay     = first matching alarm's trigger_ts - scenario.start_ts   (event time, seconds)
    per type  = the same, for each scenario type
    false positives are split into
        - "labelled_no_alarm": the alarm hits a scenario labelled as harmless
          (typo_burst, edge_exact_window), meaning the rule or the generator is wrong
        - "normal_traffic": the alarm hits no scenario at all (unlucky normal users)

Stream replay (TODO M6):
    replay events in (ts, file order) at `speedup` x real time (inf = as fast as possible);
    keep a deque of failure timestamps per user; when a new alarm opens, record
    processing latency = time of alarm - arrival time of the triggering event.
    The alarm list must equal the batch result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from login_correlation.common import EventArrays, RuleParams


def _matches(alarms: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """All (alarm, scenario) pairs that match: same user and overlapping time ranges."""
    a = alarms.reset_index(drop=True).rename_axis("alarm_idx").reset_index()
    pairs = a.merge(truth, on="user", suffixes=("", "_gt"))
    overlap = (pairs["start_ts"] <= pairs["end_ts_gt"]) & (pairs["end_ts"] >= pairs["start_ts_gt"])
    return pairs[overlap]


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def evaluate(alarms: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    """Compare alarms with the labelled scenarios. Returns a JSON-friendly dict."""
    truth = ground_truth.copy()
    truth["expected_alarm"] = truth["expected_alarm"].astype(bool)
    pairs = _matches(alarms, truth)
    expected_pairs = pairs[pairs["expected_alarm"]]

    n_alarms = len(alarms)
    good_alarms = expected_pairs["alarm_idx"].nunique()
    expected = truth[truth["expected_alarm"]]
    found_ids = set(pairs["scenario_id"])
    n_found = int(expected["scenario_id"].isin(found_ids).sum())

    precision = _ratio(good_alarms, n_alarms)
    recall = _ratio(n_found, len(expected))
    f1 = (round(2 * precision * recall / (precision + recall), 4)
          if precision and recall else None)

    # delay: first matching alarm per found scenario
    first_trigger = expected_pairs.groupby("scenario_id")["trigger_ts"].min()
    delays = (first_trigger - expected.set_index("scenario_id")["start_ts"]).dropna()

    # false positives, split by cause
    bad = set(range(n_alarms)) - set(expected_pairs["alarm_idx"])
    hits_labelled = set(pairs.loc[~pairs["expected_alarm"], "alarm_idx"]) & bad

    per_type = {}
    for kind, rows in truth.groupby("type", sort=True):
        found = int(rows["scenario_id"].isin(found_ids).sum())
        per_type[kind] = {
            "scenarios": len(rows),
            "expected_alarm": bool(rows["expected_alarm"].iloc[0]),
            "with_alarm": found,
            "share_with_alarm": _ratio(found, len(rows)),
        }

    return {
        "alarms": n_alarms,
        "true_positive_alarms": int(good_alarms),
        "false_positive_alarms": {
            "labelled_no_alarm": len(hits_labelled),
            "normal_traffic": len(bad) - len(hits_labelled),
        },
        "expected_scenarios": len(expected),
        "found_scenarios": n_found,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection_delay_s": {
            "mean": round(float(delays.mean()), 2) if len(delays) else None,
            "median": float(np.median(delays)) if len(delays) else None,
            "max": int(delays.max()) if len(delays) else None,
        },
        "per_type": per_type,
    }


def format_report(m: dict) -> str:
    """Human-readable version of evaluate()'s result, printed by `python -m login_correlation evaluate`."""
    fp = m["false_positive_alarms"]
    d = m["detection_delay_s"]
    lines = [
        f"alarms            {m['alarms']}  (true: {m['true_positive_alarms']}, "
        f"false: {fp['labelled_no_alarm'] + fp['normal_traffic']} = "
        f"{fp['labelled_no_alarm']} on harmless scenarios + {fp['normal_traffic']} from normal traffic)",
        f"attack scenarios  {m['found_scenarios']} of {m['expected_scenarios']} found",
        f"precision         {m['precision']}",
        f"recall            {m['recall']}",
        f"f1                {m['f1']}",
        f"detection delay   mean {d['mean']} s, median {d['median']} s, max {d['max']} s "
        "(from the first event of the attack to the alarm, in log time)",
        "",
        f"{'scenario type':<20}{'expected':>10}{'scenarios':>11}{'with alarm':>12}",
    ]
    for kind, r in m["per_type"].items():
        lines.append(f"{kind:<20}{'alarm' if r['expected_alarm'] else 'none':>10}"
                     f"{r['scenarios']:>11}{r['with_alarm']:>12}")
    return "\n".join(lines)


def replay(events: EventArrays, params: RuleParams, speedup: float = float("inf")) -> dict:
    """Return {"alarms": DataFrame, "latency_s": {"p50", "p95", "max"}}. TODO(M6)."""
    raise NotImplementedError("stream replay - milestone M6")
