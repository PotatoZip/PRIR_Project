"""Shared test helpers. pytest loads this file automatically; tests import from it with
`from conftest import random_events`."""

import numpy as np

from login_correlation.common import EventArrays


def random_events(seed, n=5000, n_users=50):
    """Random data that contains every edge case at once:
    many repeated timestamps (5000 events in 1 hour), 5 empty accounts, events with an
    empty user field, and an attack in the middle of the hour, so that it crosses the
    border between time chunks."""
    rng = np.random.default_rng(seed)
    ts = rng.integers(0, 3600, n).tolist()
    users = [f"user{u:03d}" for u in rng.integers(0, n_users, n)]
    result = np.where(rng.random(n) < 0.4, "FAIL", "SUCCESS").tolist()
    rows = list(zip(ts, users, ["10.0.0.1"] * n, result))
    rows += [(1790 + k, "attacker", "203.0.113.66", "FAIL") for k in range(20)]  # 1790..1809
    rows += [(100 + k, "", "198.51.100.9", "FAIL") for k in range(6)]            # empty user field
    return EventArrays.from_rows(rows, [f"user{u:03d}" for u in range(n_users + 5)])
