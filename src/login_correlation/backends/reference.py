"""Reference backend: the rule in plain Python. It defines the correct answer.

It is a direct translation of the rule in docs/ARCHITECTURE.md section 2. It deliberately
does NOT use core.py: the fast backends are checked against it, so it must not
share their code (or their bugs). It is slow; use it for up to about 1e6 events.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from login_correlation.common import (DetectionResult, EventArrays, PhaseTimer, RuleParams,
                            make_alarm_id, normalize_alarms, normalize_summary)


def detect(events: EventArrays, params: RuleParams, **_: object) -> DetectionResult:
    timer = PhaseTimer()
    W, N = params.window_s, params.threshold

    with timer.phase("total"):
        # 1. Group failure timestamps per user. Rows are in file order (seq) and Python's
        #    sort is stable, so equal timestamps keep their file order.
        with timer.phase("prepare"):
            fails: dict[int, list[int]] = defaultdict(list)
            n_events, n_fail = defaultdict(int), defaultdict(int)
            first_ts: dict[int, int] = {}
            last_ts: dict[int, int] = {}
            for ts, uid, f in zip(events.ts.tolist(), events.user_id.tolist(),
                                  events.is_fail.tolist()):
                n_events[uid] += 1
                first_ts[uid] = min(first_ts.get(uid, ts), ts)
                last_ts[uid] = max(last_ts.get(uid, ts), ts)
                if f:
                    n_fail[uid] += 1
                    fails[uid].append(ts)
            for lst in fails.values():
                lst.sort()

        # 2. Sliding window + merging of overlapping triggers into one alarm.
        with timer.phase("compute"):
            alarms, max_count = [], {}
            for uid, f in fails.items():
                user, j, best, cur = events.users[uid], 0, 0, None   # cur = [first_j, first_i, last_i]
                for i, t in enumerate(f):
                    while f[j] <= t - W:          # failures at or before t - W are outside
                        j += 1
                    count = i - j + 1
                    best = max(best, count)
                    if count >= N:
                        if cur is not None and j <= cur[2]:
                            cur[2] = i            # window shares a failure with the open alarm
                        else:
                            if cur is not None:
                                alarms.append(_alarm(user, f, cur))
                            cur = [j, i, i]
                if cur is not None:
                    alarms.append(_alarm(user, f, cur))
                max_count[uid] = best

        # 3. Per-account summary (one row for every account, also the empty ones).
        with timer.phase("merge"):
            alarms_df = normalize_alarms(pd.DataFrame(alarms))
            per_user = alarms_df["user"].value_counts().to_dict()
            rows = [{
                "user": user,
                "n_events": n_events.get(uid, 0),
                "n_fail": n_fail.get(uid, 0),
                "n_success": n_events.get(uid, 0) - n_fail.get(uid, 0),
                "first_ts": first_ts.get(uid),
                "last_ts": last_ts.get(uid),
                "max_window_count": max_count.get(uid, 0),
                "n_alarms": int(per_user.get(user, 0)),
            } for uid, user in enumerate(events.users)]
            summary_df = normalize_summary(pd.DataFrame(rows))

    return DetectionResult(alarms_df, summary_df, timer.as_dict())


def _alarm(user: str, f: list[int], ep: list[int]) -> dict:
    first_j, first_i, last_i = ep
    return {"alarm_id": make_alarm_id(user, f[first_j]), "user": user,
            "start_ts": f[first_j], "end_ts": f[last_i], "trigger_ts": f[first_i],
            "n_failures": last_i - first_j + 1}
