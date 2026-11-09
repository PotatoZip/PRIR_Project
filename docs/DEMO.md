# Demonstration guide

Step-by-step instructions for presenting the project on the Windows 11 laptop.
Setup and configuration are in the main [README](../README.md).
This guide grows with every milestone. Steps marked ⬜ are not written yet; if you run
them, they print "not implemented yet".

All commands run in PowerShell, in the project folder, with the environment active
(`.\.venv\Scripts\Activate.ps1`). The whole demonstration can also be run in one go with
`.\demo.ps1` (use `.\demo.ps1 -Events 200000` for a quick run).

| # | Step | Shows | Status |
|---|---|---|---|
| 0 | Environment check | OpenMP, MPI and OpenCL work on this laptop | ✅ |
| 1 | Tests | the rule's edge cases are handled correctly | ✅ |
| 2 | Data generation | labelled attacks hidden in normal traffic | ✅ M1 |
| 3 | Detection: reference | the alarm list and the per-account summary | ✅ |
| 4 | Detection quality | precision and recall | ⬜ M2 |
| 5 | OpenMP | split by accounts and by time + halo, speed-up | ⬜ M3 |
| 6 | MPI | processes, one owner per alarm | ⬜ M4 |
| 7 | OpenCL | GPU pipeline, encoding cost shown separately | ⬜ M5 |
| 8 | Same results everywhere | every backend = reference | ⬜ M3–M5 |
| 9 | Stream replay | time-to-alarm | ⬜ M6 |
| 10 | Performance charts | events/s, speed-up | ⬜ M7 |

---

## 0. Environment check

```powershell
python -m login_correlation check
```
Expected: every line `[ OK ]`, ending with `All good.` Point out:
* `OpenMP (Numba)  layer=omp, threads=12`: Python code runs on real OpenMP threads.
* `mpiexec -n 2  2 processes started`: MS-MPI works.
* `OpenCL kernel run  Intel(R) Iris(R) Xe Graphics`: the GPU runs our code.

## 1. Tests

```powershell
pytest -q
```
Expected: all tests pass. Tests for backends that are not written yet are reported as "skipped".
Worth showing: `tests/test_rule.py`. Its test names read like the edge cases from the task
(`test_event_exactly_W_before_is_outside_window`, `test_repeated_timestamps`,
`test_empty_account_is_in_summary`, …).

## 2. Data generation ✅

```powershell
python -m login_correlation generate --out data\demo                  # 1 M events, about 5 s
python -m login_correlation generate --out data\d10m --events 10000000   # 10 M events, about 1 min, 400 MB
```
Expected output (1 M):
```
generated 1,001,506 events (1,024 in 113 scenarios), 10,000 users + 50 empty accounts in 5.1 s
```

Show the three files:
```powershell
Get-Content data\demo\events.csv -TotalCount 5        # ts,user,src_ip,result
Import-Csv data\demo\ground_truth.csv | Format-Table  # the labelled scenarios
Import-Csv data\demo\ground_truth.csv | Group-Object type | Select-Object Name, Count
```

What to explain:
* **Normal traffic:** a few very active users and many quiet ones. Each user logs in from
  1–3 home IPs (10.x.x.x), and about 3 % of logins start with 1–2 typos.
* **Labelled scenarios** (`ground_truth.csv`, which the detector never reads):

  | Type | What it is | Expected |
  |---|---|---|
  | `brute_force` | 5–25 failures within 60 s from an attacker IP (sometimes followed by a successful login) | alarm |
  | `slow_brute` | failures spread out below the rule's rate | an attack, but the rule **misses** it (known limit) |
  | `typo_burst` | 4 typos, then a successful login | no alarm |
  | `edge_exact_window` | 5 failures, first and last exactly 60 s apart | no alarm (the edge is outside the window) |
  | `edge_inside_window` | 5 failures within 59 s | alarm |
  | `same_timestamp` | 5–10 failures in the same second | alarm |
  | `empty_username` | failures with an empty user field | alarm for account `<empty>` |
  | (empty accounts) | 50 accounts in `users.csv` with no events | listed in the summary with zeros |

* Each victim's own normal traffic is removed around its scenario, so the label is
  guaranteed to be correct.
* Repeatable: the same `--seed` gives byte-identical files (this is tested).

## 3. Detection with the reference backend ✅

```powershell
python -m login_correlation detect --data data\sample --out results\sample   # 14 events, easy to read
Get-Content results\sample\alarms.csv
Get-Content results\sample\summary.csv

python -m login_correlation detect --data data\demo --out results\demo_ref   # 1 M events
```
Expected on `data\sample`: one alarm, for **bob** (6 failures in 15 s). **carol** has 5
failures 30 s apart, which is too slow for W=60/N=5, so there is no alarm. **dave** is an
empty account: he is in the summary with zeros.

Expected on `data\demo` (seed 42, W=60, N=5), measured on this laptop:
* 65 alarms. 63 of them match labelled attacks.
* All `brute_force`, `edge_inside_window`, `same_timestamp` and `empty_username`
  scenarios are found.
* No alarm for `typo_burst` or `edge_exact_window`. All `slow_brute` are missed, as designed.
* 2 alarms come from very active normal users who made typos close together. These are
  realistic false positives, and a good talking point about precision.

Try other rule settings: `--threshold 3` or `--window 300`.

## 4–10. Coming with the next milestones ⬜
These sections are filled in as each milestone is finished (see [ROADMAP.md](ROADMAP.md)).
