"""Measuring the results: detection quality and stream replay

Quality (docs/ARCHITECTURE.md section 8):
    an alarm matches a scenario if it is the same user and the time ranges overlap
    precision = alarms matching an expected scenario / all alarms
    recall    = expected scenarios found by an alarm / all expected scenarios
    delay     = alarm.trigger_ts - scenario.start_ts
    also: results per scenario type

Stream replay:
    replay events in (ts, file order) at `speedup` x real time (inf = as fast as possible);
    keep a deque of failure timestamps per user; when a new alarm opens, record
    processing latency = time of alarm - arrival time of the triggering event.
    The alarm list must equal the batch result.
"""

from __future__ import annotations

import pandas as pd

from login_correlation.common import EventArrays, RuleParams


def evaluate(alarms: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    """Return {"precision", "recall", "per_type", "delay_s"}. TODO(M2)."""
    raise NotImplementedError("evaluate - milestone M2")


def replay(
    events: EventArrays, params: RuleParams, speedup: float = float("inf")
) -> dict:
    """Return {"alarms": DataFrame, "latency_s": {"p50", "p95", "max"}}. TODO(M6)."""
    raise NotImplementedError("stream replay - milestone M6")
