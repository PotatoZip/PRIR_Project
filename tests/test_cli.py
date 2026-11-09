"""Command line and config file, run on data/sample (its accounts are described in docs/DEMO.md, step 3)."""

from pathlib import Path

import pandas as pd

from login_correlation.__main__ import main
from login_correlation.common import load_config

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_config_file_is_read():
    cfg = load_config()
    assert cfg["rule"] == {"window_s": 60, "threshold": 5}


def test_detect_reference_on_sample(tmp_path):
    assert main(["detect", "--data", str(SAMPLE), "--out", str(tmp_path)]) == 0
    alarms = pd.read_csv(tmp_path / "alarms.csv")
    assert list(alarms["user"]) == ["bob"]           # carol is too slow -> no alarm
    assert alarms.iloc[0]["n_failures"] == 6
    summary = pd.read_csv(tmp_path / "summary.csv").set_index("user")
    assert list(summary.index) == ["alice", "bob", "carol", "dave"]
    assert summary.loc["dave", "n_events"] == 0     # empty account
    assert (tmp_path / "timings.json").exists()


def test_command_line_overrides_config(tmp_path):
    # with N = 2 carol's slow attack (2 failures in every 60 s window) is found too
    assert main(["detect", "--data", str(SAMPLE), "--out", str(tmp_path), "--threshold", "2"]) == 0
    assert set(pd.read_csv(tmp_path / "alarms.csv")["user"]) == {"bob", "carol"}


def test_compare(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    main(["detect", "--data", str(SAMPLE), "--out", str(a)])
    main(["detect", "--data", str(SAMPLE), "--out", str(b)])
    assert main(["compare", str(a), str(b)]) == 0
