"""Edge cases of the rule (docs/ARCHITECTURE.md section 2), checked on the reference backend.

These tests define what "correct" means for the whole project.
"""

import pytest

from login_correlation.backends.reference import detect
from login_correlation.common import EMPTY_USER, EventArrays, RuleParams

W, N = 60, 3
P = RuleParams(window_s=W, threshold=N)


def fails(user, *ts):
    return [(t, user, "10.0.0.1", "FAIL") for t in ts]


def run(rows, registry=(), params=P):
    return detect(EventArrays.from_rows(rows, registry), params)


def test_empty_account_is_in_summary():
    res = run([], registry=["alice"])
    assert res.alarms.empty
    row = res.summary.iloc[0]
    assert (row["user"], row["n_events"], row["max_window_count"]) == ("alice", 0, 0)


def test_event_exactly_W_before_is_outside_window():
    res = run(fails("u", 0, 30, 60))          # first and last exactly W apart
    assert res.alarms.empty
    assert res.summary.iloc[0]["max_window_count"] == 2


def test_event_W_minus_1_before_is_inside_window():
    a = run(fails("u", 0, 30, 59)).alarms
    assert len(a) == 1
    assert tuple(a.iloc[0][["start_ts", "end_ts", "trigger_ts", "n_failures"]]) == (0, 59, 59, 3)


def test_repeated_timestamps():
    a = run(fails("u", 100, 100, 100)).alarms
    assert len(a) == 1 and a.iloc[0]["n_failures"] == 3


def test_success_neither_counts_nor_resets():
    res = run(fails("u", 0, 10) + [(15, "u", "10.0.0.1", "SUCCESS")] + fails("u", 20))
    assert len(res.alarms) == 1
    assert tuple(res.summary.iloc[0][["n_events", "n_fail", "n_success"]]) == (4, 3, 1)


def test_overlapping_triggers_make_one_alarm():
    a = run(fails("u", *range(10))).alarms     # 10 failures within 10 s
    assert len(a) == 1
    assert tuple(a.iloc[0][["start_ts", "end_ts", "trigger_ts", "n_failures"]]) == (0, 9, 2, 10)


def test_separate_bursts_make_two_alarms():
    assert list(run(fails("u", 0, 1, 2) + fails("u", 1000, 1001, 1002)).alarms["start_ts"]) == [0, 1000]


def test_users_are_independent():
    assert run(fails("a", 0, 1) + fails("b", 2)).alarms.empty


def test_threshold_one():
    assert len(run(fails("u", 0, 500), params=RuleParams(W, 1)).alarms) == 2


def test_file_order_does_not_matter():
    assert run(fails("u", 59, 0, 30)).alarms.equals(run(fails("u", 0, 30, 59)).alarms)


def test_empty_username_is_its_own_account():
    rows = fails("", 0, 1, 2) + fails("bob", 3)
    res = run(rows)
    assert list(res.alarms["user"]) == [EMPTY_USER]
    assert EMPTY_USER in set(res.summary["user"])


@pytest.mark.parametrize("bad", [dict(window_s=0), dict(threshold=0)])
def test_invalid_params(bad):
    with pytest.raises(ValueError):
        RuleParams(**bad)
