"""Synthetic login traffic with labelled attack scenarios

Writes to <out_dir> (formats in docs/ARCHITECTURE.md section 5):
    events.csv        ts,user,src_ip,result      sorted by ts - the detector input
    users.csv         user                       includes n_empty_users accounts with no events
    ground_truth.csv  scenario_id,type,user,start_ts,end_ts,n_events,expected_alarm

Suggested approach (NumPy, so 10M events take seconds):
  1. rng = np.random.default_rng(seed)
  2. normal traffic: user picked with probability ~ 1/rank (a few very active users);
     ts = start_ts + rng.integers(0, duration_s, n); each user has 1-3 home IPs;
     a share `typo_rate` of logins gets 1-2 FAILs a few seconds before the SUCCESS
  3. scenarios (table in docs/ARCHITECTURE.md section 8): for each one pick a fresh victim
     account that has no normal traffic, so the expected result is certain
     empty_username: N failures with an empty user field (written as an empty CSV value)
  4. concatenate, stable-sort by ts, write the three CSV files
"""

from __future__ import annotations

from pathlib import Path

from login_correlation.common import GeneratorParams, RuleParams


def generate(params: GeneratorParams, rule: RuleParams, out_dir: str | Path) -> Path:
    """TODO(M1)."""
    raise NotImplementedError("Not yet implemented")
