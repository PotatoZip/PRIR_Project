"""Detection-quality metrics (evaluate.py, milestone M2): precision, recall, delay, per type."""

import json

import pandas as pd
import pytest

from login_correlation.__main__ import main
from login_correlation.common import GeneratorParams, RuleParams
from login_correlation.evaluate import evaluate, format_report
from login_correlation.generator import SCENARIO_TYPES, generate

TRUTH = pd.DataFrame(
    [  # scenario_id, type, user, start_ts, end_ts, n_events, expected_alarm
        ("S1", "brute_force", "alice", 100, 130, 8, True),
        ("S2", "brute_force", "bob", 500, 520, 6, True),
        ("S3", "slow_brute", "carol", 1000, 1600, 10, True),
        ("S4", "typo_burst", "dave", 2000, 2008, 5, False),
    ],
    columns=["scenario_id", "type", "user", "start_ts", "end_ts", "n_events", "expected_alarm"],
)


def alarms(*rows):
    """rows: (user, start_ts, end_ts, trigger_ts)"""
    return pd.DataFrame(
        [(f"{u}:{s}", u, s, e, t, 5) for u, s, e, t in rows],
        columns=["alarm_id", "user", "start_ts", "end_ts", "trigger_ts", "n_failures"],
    )


def test_perfect_result():
    m = evaluate(alarms(("alice", 100, 130, 110), ("bob", 500, 520, 515), ("carol", 1000, 1600, 1500)), TRUTH)
    assert (m["precision"], m["recall"], m["f1"]) == (1.0, 1.0, 1.0)
    # delays: 110-100 = 10, 515-500 = 15, 1500-1000 = 500
    assert m["detection_delay_s"] == {"mean": 175.0, "median": 15.0, "max": 500}


def test_missed_attack_lowers_recall():
    m = evaluate(alarms(("alice", 100, 130, 110), ("bob", 500, 520, 515)), TRUTH)
    assert m["precision"] == 1.0
    assert m["recall"] == round(2 / 3, 4)
    assert m["per_type"]["slow_brute"]["with_alarm"] == 0


def test_false_positive_on_harmless_scenario():
    m = evaluate(alarms(("alice", 100, 130, 110), ("dave", 2000, 2008, 2008)), TRUTH)
    assert m["precision"] == 0.5
    assert m["false_positive_alarms"] == {"labelled_no_alarm": 1, "normal_traffic": 0}
    assert m["per_type"]["typo_burst"]["with_alarm"] == 1


def test_false_positive_from_normal_traffic():
    m = evaluate(alarms(("alice", 100, 130, 110), ("eve", 5, 9, 9)), TRUTH)
    assert m["false_positive_alarms"] == {"labelled_no_alarm": 0, "normal_traffic": 1}


def test_wrong_time_or_user_does_not_match():
    m = evaluate(alarms(("alice", 131, 200, 150), ("bob", 100, 130, 110)), TRUTH)
    assert m["true_positive_alarms"] == 0
    assert m["recall"] == 0.0


@pytest.mark.parametrize("start, end", [(130, 140), (90, 100), (90, 140), (110, 120)])
def test_touching_or_overlapping_ranges_match(start, end):
    m = evaluate(alarms(("alice", start, end, end)), TRUTH)
    assert m["found_scenarios"] == 1


def test_one_alarm_counts_once_even_if_it_matches_twice():
    truth = pd.concat([TRUTH, TRUTH.iloc[[0]].assign(scenario_id="S5", start_ts=120, end_ts=150)])
    m = evaluate(alarms(("alice", 100, 150, 110)), truth)
    assert m["true_positive_alarms"] == 1 and m["precision"] == 1.0
    assert m["found_scenarios"] == 2


def test_no_alarms():
    m = evaluate(alarms(), TRUTH)
    assert m["precision"] is None and m["recall"] == 0.0 and m["f1"] is None
    assert m["detection_delay_s"]["mean"] is None
    format_report(m)  # must not crash on empty values


def test_result_is_json_serialisable():
    json.dumps(evaluate(alarms(("alice", 100, 130, 110)), TRUTH))


def test_cli_evaluate_on_generated_data(tmp_path, capsys):
    data, out = tmp_path / "data", tmp_path / "out"
    params = GeneratorParams(seed=5, n_users=300, n_events=20_000, duration_s=24 * 3600,
                             scenarios={t: 3 for t in SCENARIO_TYPES})
    generate(params, RuleParams(60, 5), data)
    assert main(["detect", "--data", str(data), "--out", str(out)]) == 0
    assert main(["evaluate", "--data", str(data), "--results", str(out)]) == 0
    m = json.loads((out / "metrics.json").read_text())
    # every attack is found except slow_brute (built to stay below the rule's rate)
    assert m["found_scenarios"] == m["expected_scenarios"] - 3
    assert m["per_type"]["slow_brute"]["with_alarm"] == 0
    assert m["false_positive_alarms"]["labelled_no_alarm"] == 0
    assert "precision" in capsys.readouterr().out
