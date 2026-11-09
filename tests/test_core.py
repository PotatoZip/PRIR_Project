"""The fast compiled core (core.py, milestone M2) must give exactly the reference result.

The rule's edge cases are already run on core in test_rule.py. This file checks the
building blocks one by one, and then the whole pipeline on random and generated data.
"""

import numpy as np
import pytest
from conftest import random_events

from login_correlation import core
from login_correlation.backends import reference
from login_correlation.common import EventArrays, GeneratorParams, RuleParams, encode, load_dataset
from login_correlation.generator import SCENARIO_TYPES, generate

# ----------------------------------------------------------------- building blocks


@pytest.mark.parametrize("value, expected", [(0, 0), (1, 0), (2, 1), (3, 1), (5, 3), (9, 5), (10, 5)])
def test_lower_bound(value, expected):
    a = np.array([1, 3, 3, 5, 7], dtype=np.int64)
    assert core.lower_bound(a, 0, 5, value) == expected


def test_lower_bound_respects_range():
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    assert core.lower_bound(a, 2, 4, 0) == 2     # never below lo
    assert core.lower_bound(a, 2, 4, 99) == 4    # never above hi


def test_sort_failures_groups_users_and_keeps_tie_order():
    ev = EventArrays.from_rows(
        [(5, "b", "ip", "FAIL"), (3, "a", "ip", "FAIL"), (3, "a", "ip", "SUCCESS"),
         (1, "b", "ip", "FAIL"), (3, "a", "ip", "FAIL")],
        registry=["a", "b", "c"],
    )
    ts, off = core.sort_failures(ev)
    assert ts.tolist() == [3, 3, 1, 5]        # a: 3, 3   b: 1, 5   (successes removed)
    assert off.tolist() == [0, 2, 4, 4]       # c has no failures -> empty segment


def test_window_counts_by_hand():
    # one user, W = 10: windows are (t-10, t]
    ts = np.array([0, 5, 9, 10, 10, 25], dtype=np.int64)
    off = np.array([0, 6], dtype=np.int64)
    j, count = core.window_counts(ts, off, 10, np.array([0], dtype=np.int64))
    assert count.tolist() == [1, 2, 3, 3, 4, 1]   # at t=10 the failure at t=0 is outside
    assert j.tolist() == [0, 0, 0, 1, 1, 5]


def test_window_counts_do_not_cross_users():
    ts = np.array([0, 1, 2, 2, 3], dtype=np.int64)   # user 0: 0,1,2   user 1: 2,3
    off = np.array([0, 3, 5], dtype=np.int64)
    _, count = core.window_counts(ts, off, 100, np.array([0, 1], dtype=np.int64))
    assert count.tolist() == [1, 2, 3, 1, 2]


def test_user_order_does_not_change_counts():
    ts, off = core.sort_failures(random_events(3))
    U = off.shape[0] - 1
    a = core.window_counts(ts, off, 60, np.arange(U, dtype=np.int64))
    b = core.window_counts(ts, off, 60, np.arange(U, dtype=np.int64)[::-1].copy())
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_find_episodes_merges_overlapping_triggers():
    # 10 failures 1 s apart, then a gap, then 3 more; N = 3, W = 60
    ts = np.array(list(range(10)) + [500, 501, 502], dtype=np.int64)
    off = np.array([0, 13], dtype=np.int64)
    j, count = core.window_counts(ts, off, 60, np.array([0], dtype=np.int64))
    ep_user, first_j, first_i, last_i, max_count = core.find_episodes(off, j, count, 3)
    assert ep_user.tolist() == [0, 0]
    assert ts[first_j].tolist() == [0, 500]       # start_ts
    assert ts[first_i].tolist() == [2, 502]       # trigger_ts
    assert ts[last_i].tolist() == [9, 502]        # end_ts
    assert max_count.tolist() == [10]


# ----------------------------------------------------------------- whole pipeline = reference


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("params", [RuleParams(60, 5), RuleParams(10, 2), RuleParams(300, 10),
                                    RuleParams(1, 1), RuleParams(3600, 50)])
def test_core_equals_reference_on_random_data(seed, params):
    events = random_events(seed)
    expected = reference.detect(events, params)
    got = core.detect(events, params)
    assert got.alarms.equals(expected.alarms)
    assert got.summary.equals(expected.summary)


def test_core_equals_reference_on_generated_data(tmp_path):
    rule = RuleParams(60, 5)
    params = GeneratorParams(seed=11, n_users=300, n_events=30_000, duration_s=24 * 3600,
                             scenarios={t: 3 for t in SCENARIO_TYPES})
    generate(params, rule, tmp_path)
    df, registry = load_dataset(tmp_path)
    events = encode(df, registry)
    expected = reference.detect(events, rule)
    got = core.detect(events, rule)
    assert got.alarms.equals(expected.alarms)
    assert got.summary.equals(expected.summary)
    assert len(got.alarms) > 0


def test_no_failures_at_all():
    ev = EventArrays.from_rows([(1, "a", "ip", "SUCCESS")], registry=["a", "b"])
    res = core.detect(ev, RuleParams(60, 1))
    assert res.alarms.equals(reference.detect(ev, RuleParams(60, 1)).alarms)
    assert res.alarms.empty


def test_timings_have_the_standard_phases():
    res = core.detect(random_events(0), RuleParams(60, 5))
    assert {"prepare", "compute", "merge", "total"} <= set(res.timings)


def test_warm_up_returns_seconds():
    assert core.warm_up() >= 0
