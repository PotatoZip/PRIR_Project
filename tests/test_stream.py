"""Stream replay and time-to-alarm (stream.py, milestone M6).

The replayed stream must raise exactly the alarms of batch detection, in every edge case,
and the time measurements must be consistent with the replay speed.
"""

import json
import math
from pathlib import Path

import pytest
from conftest import random_events

from login_correlation.__main__ import main
from login_correlation.backends import reference
from login_correlation.common import EventArrays, RuleParams
from login_correlation.stream import replay

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"
P = RuleParams(60, 3)


def fails(user, *ts):
    return [(t, user, "10.0.0.1", "FAIL") for t in ts]


def same_as_batch(ev, params):
    return replay(ev, params)["alarms"].equals(reference.detect(ev, params).alarms)


# ----------------------------------------------------------------- same alarms as batch


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("params", [RuleParams(60, 5), RuleParams(10, 2), RuleParams(300, 10),
                                    RuleParams(1, 1), RuleParams(3600, 50)])
def test_replay_equals_batch_on_random_data(seed, params):
    assert same_as_batch(random_events(seed), params)


@pytest.mark.parametrize("rows, alarms", [
    (fails("u", 0, 30, 60), 0),                  # exactly W apart: outside the window
    (fails("u", 0, 30, 59), 1),                  # W - 1 apart: alarm
    (fails("u", 100, 100, 100), 1),              # equal timestamps
    (fails("u", *range(10)), 1),                 # overlapping triggers -> one alarm
    (fails("u", 0, 1, 2) + fails("u", 1000, 1001, 1002), 2),   # two separate attacks
    (fails("", 5, 6, 7), 1),                     # empty user field
    (fails("u", 59, 0, 30), 1),                  # file not in time order: replay sorts by time
])
def test_edge_cases(rows, alarms):
    ev = EventArrays.from_rows(rows, registry=["nobody"])
    res = replay(ev, P)
    assert len(res["alarms"]) == alarms
    assert res["alarms"].equals(reference.detect(ev, P).alarms)


def test_success_does_not_reset_the_window():
    ev = EventArrays.from_rows(fails("u", 0, 10) + [(15, "u", "ip", "SUCCESS")] + fails("u", 20))
    assert len(replay(ev, P)["alarms"]) == 1


def test_no_events_and_no_failures():
    assert replay(EventArrays.from_rows([], registry=["a"]), P)["alarms"].empty
    ev = EventArrays.from_rows([(1, "a", "ip", "SUCCESS")])
    res = replay(ev, P)
    assert res["alarms"].empty and res["stats"]["alarms"] == 0
    assert res["stats"]["processing_latency_ms"]["p50"] is None


# ----------------------------------------------------------------- the alarm is raised at the trigger


def test_alarm_is_raised_at_the_nth_failure():
    ev = EventArrays.from_rows(fails("u", 100, 110, 125, 130, 140))   # N = 3 -> trigger at 125
    res = replay(ev, P)
    row = res["per_alarm"].iloc[0]
    assert (row["start_ts"], row["trigger_ts"], row["detection_delay_s"]) == (100, 125, 25)
    assert len(res["per_alarm"]) == 1        # later failures extend the alarm, no new one


def test_one_raised_alarm_per_final_alarm():
    res = replay(random_events(1), RuleParams(30, 3))
    assert len(res["per_alarm"]) == len(res["alarms"])
    assert (res["per_alarm"]["trigger_ts"].sort_values().tolist()
            == res["alarms"]["trigger_ts"].sort_values().tolist())


# ----------------------------------------------------------------- timing


def test_unpaced_replay_reports_fast_processing():
    s = replay(random_events(0), RuleParams(60, 5))["stats"]
    assert s["speedup"] == "inf" and s["max_lag_ms"] is None
    assert s["processing_latency_ms"]["max"] < 100          # well under 0.1 s per alarm
    assert s["events_per_s"] > 10_000


def test_paced_replay_follows_the_clock():
    # 20 s of log time at speed-up 100 -> about 0.2 s of real time
    ev = EventArrays.from_rows(fails("u", 0, 5, 10) + [(20, "x", "ip", "SUCCESS")])
    res = replay(ev, P, speedup=100)
    s = res["stats"]
    assert 0.18 <= s["wall_s"] < 0.5
    assert s["max_lag_ms"] < 50
    # time to alarm (real) = detection delay (log) / speed-up + processing  -> about 0.1 s
    tta = res["per_alarm"].iloc[0]["time_to_alarm_s"]
    assert 0.09 <= tta < 0.2


def test_invalid_speedup():
    for bad in (0, -5):
        with pytest.raises(ValueError):
            replay(random_events(0), P, speedup=bad)
    assert replay(random_events(0), P, speedup=math.inf)["stats"]["speedup"] == "inf"


# ----------------------------------------------------------------- command line


def test_cli_replay_on_sample(tmp_path, capsys):
    assert main(["replay", "--data", str(SAMPLE), "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "same alarms as batch detection: YES" in out
    stats = json.loads((tmp_path / "replay.json").read_text())
    assert stats["alarms"] == 1 and stats["same_as_batch"] is True
    assert (tmp_path / "alarms.csv").exists() and (tmp_path / "per_alarm.csv").exists()


def test_cli_replay_alarms_equal_detect(tmp_path):
    ref, rep = tmp_path / "ref", tmp_path / "rep"
    assert main(["detect", "--data", str(SAMPLE), "--out", str(ref)]) == 0
    assert main(["replay", "--data", str(SAMPLE), "--out", str(rep), "--threshold", "5"]) == 0
    assert (ref / "alarms.csv").read_bytes() == (rep / "alarms.csv").read_bytes()


def test_compare_works_with_replay_output(tmp_path, capsys):
    """`replay` writes no summary.csv; `compare` then checks the alarms only."""
    ref, rep = tmp_path / "ref", tmp_path / "rep"
    assert main(["detect", "--data", str(SAMPLE), "--out", str(ref)]) == 0
    assert main(["replay", "--data", str(SAMPLE), "--out", str(rep)]) == 0
    assert main(["compare", str(ref), str(rep)]) == 0
    out = capsys.readouterr().out
    assert "alarms.csv: IDENTICAL" in out and "summary.csv: skipped" in out


def test_bench_stream_latency_on_sample(tmp_path):
    """The benchmark code (bench.py) replays at several speeds and draws the chart."""
    from login_correlation import bench

    summary = bench.stream_latency(SAMPLE, [math.inf, 10000])      # 305 s of logs -> ~0.03 s
    assert summary["speedup"].tolist() == ["inf", "10000x"]
    assert (summary["alarms"] == 1).all()
    bench.plot_stream(summary, tmp_path / "stream.png")
    assert (tmp_path / "stream.png").stat().st_size > 0
