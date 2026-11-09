"""Tests of the data generator (milestone M1).

They check the file format, that the output is repeatable, and, most importantly, that
every labelled scenario really behaves as its label says when the reference detector
runs on it.
"""

import pandas as pd
import pytest

from login_correlation.backends.reference import detect
from login_correlation.common import EMPTY_USER, GeneratorParams, RuleParams, encode, load_dataset
from login_correlation.generator import SCENARIO_TYPES, generate

RULE = RuleParams(window_s=60, threshold=5)
SMALL = GeneratorParams(seed=7, n_users=500, n_events=20_000, duration_s=24 * 3600, n_empty_users=10,
                        scenarios={t: 4 for t in SCENARIO_TYPES})


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    generate(SMALL, RULE, out)
    events = pd.read_csv(out / "events.csv", keep_default_na=False)   # keep "" as ""
    users = pd.read_csv(out / "users.csv")
    truth = pd.read_csv(out / "ground_truth.csv")
    return out, events, users, truth


@pytest.fixture(scope="module")
def alarms(dataset):
    out = dataset[0]
    df, registry = load_dataset(out)
    return detect(encode(df, registry), RULE).alarms


# --------------------------------------------------------------------- format

def test_files_and_columns(dataset):
    out, events, users, truth = dataset
    assert list(events.columns) == ["ts", "user", "src_ip", "result"]
    assert list(users.columns) == ["user"]
    assert list(truth.columns) == ["scenario_id", "type", "user", "start_ts", "end_ts",
                                   "n_events", "expected_alarm"]


def test_events_sorted_and_valid(dataset):
    _, events, _, _ = dataset
    assert events["ts"].is_monotonic_increasing
    assert set(events["result"]) == {"FAIL", "SUCCESS"}
    assert events["ts"].min() >= SMALL.start_ts
    assert events["ts"].max() < SMALL.start_ts + SMALL.duration_s


def test_size_is_close_to_requested(dataset):
    _, events, _, _ = dataset
    assert abs(len(events) - SMALL.n_events) < 0.05 * SMALL.n_events


def test_repeated_timestamps_exist(dataset):
    _, events, _, _ = dataset
    assert events["ts"].duplicated().any()


# --------------------------------------------------------------------- repeatability

def test_same_seed_gives_identical_files(tmp_path):
    generate(SMALL, RULE, tmp_path / "a")
    generate(SMALL, RULE, tmp_path / "b")
    for name in ("events.csv", "users.csv", "ground_truth.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_different_seed_gives_different_events(tmp_path):
    other = GeneratorParams(**{**SMALL.__dict__, "seed": 8})
    generate(SMALL, RULE, tmp_path / "a")
    generate(other, RULE, tmp_path / "b")
    assert (tmp_path / "a" / "events.csv").read_bytes() != (tmp_path / "b" / "events.csv").read_bytes()


# --------------------------------------------------------------------- accounts

def test_empty_accounts_are_in_registry_without_events(dataset):
    _, events, users, _ = dataset
    empty = [f"user{i:06d}" for i in range(SMALL.n_users, SMALL.n_users + SMALL.n_empty_users)]
    assert set(empty) <= set(users["user"])
    assert not set(empty) & set(events["user"])


def test_empty_username_events_exist(dataset):
    _, events, _, truth = dataset
    assert (events["user"] == "").sum() > 0
    assert set(truth.loc[truth["type"] == "empty_username", "user"]) == {EMPTY_USER}


# --------------------------------------------------------------------- scenarios

def test_every_scenario_type_present(dataset):
    _, _, _, truth = dataset
    assert truth["type"].value_counts().to_dict() == {t: 4 for t in SCENARIO_TYPES}
    assert truth["scenario_id"].is_unique


def test_victims_are_different_accounts(dataset):
    _, _, _, truth = dataset
    named = truth[truth["user"] != EMPTY_USER]
    assert named["user"].is_unique


def test_edge_scenarios_have_exact_spans(dataset):
    _, _, _, truth = dataset
    span = truth["end_ts"] - truth["start_ts"]
    W = RULE.window_s
    assert (span[truth["type"] == "edge_exact_window"] == W).all()
    assert (span[truth["type"] == "edge_inside_window"] == W - 1).all()
    assert (span[truth["type"] == "same_timestamp"] == 0).all()


def _overlaps(alarms, row):
    a = alarms[alarms["user"] == row["user"]]
    return ((a["start_ts"] <= row["end_ts"]) & (a["end_ts"] >= row["start_ts"])).any()


@pytest.mark.parametrize("kind", ["brute_force", "edge_inside_window", "same_timestamp", "empty_username"])
def test_attacks_are_detected(dataset, alarms, kind):
    truth = dataset[3]
    rows = truth[truth["type"] == kind]
    assert all(_overlaps(alarms, r) for _, r in rows.iterrows())


@pytest.mark.parametrize("kind", ["typo_burst", "edge_exact_window", "slow_brute"])
def test_non_alarms_stay_silent(dataset, alarms, kind):
    """typo_burst and edge_exact_window are not attacks. slow_brute is an attack, but it is
    built to stay below the rule's rate, so the rule must miss it (a known false negative)."""
    truth = dataset[3]
    rows = truth[truth["type"] == kind]
    assert not any(_overlaps(alarms, r) for _, r in rows.iterrows())


def test_threshold_one_skips_impossible_scenarios(tmp_path):
    generate(SMALL, RuleParams(60, 1), tmp_path)
    types = set(pd.read_csv(tmp_path / "ground_truth.csv")["type"])
    assert types == {"brute_force", "same_timestamp", "empty_username"}


def test_too_many_scenarios_is_an_error(tmp_path):
    tiny = GeneratorParams(n_users=3, n_events=100, scenarios={"brute_force": 5})
    with pytest.raises(ValueError):
        generate(tiny, RULE, tmp_path)
