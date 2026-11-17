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
| 4 | Detection quality | precision and recall | ✅ M2 |
| 5 | OpenMP | split by accounts and by time + halo, speed-up | ✅ M3 |
| 6 | MPI | processes, one owner per alarm | ✅ M4 |
| 7 | OpenCL | GPU pipeline, encoding cost shown separately | ✅ M5 |
| 8 | Same results everywhere | every backend = reference | ✅ (steps 5-7 each end with `compare`) |
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
python -m login_correlation generate --out data\demo                                    # 1 M events, about 4 s
python -m login_correlation generate --out data\d10m --events 10000000 --users 100000   # 10 M events, about 1 min, 400 MB
```
Expected output (1 M):
```
generated 1,001,504 events (1,024 in 113 scenarios), 10,000 users + 50 empty accounts in 4.2 s
```
For bigger datasets, raise `--users` too (about 100 events per user). Otherwise each user
logs in unrealistically often.

Show the three files:
```powershell
Get-Content data\demo\events.csv -TotalCount 5        # ts,user,src_ip,result
Import-Csv data\demo\ground_truth.csv | Format-Table  # the labelled scenarios
Import-Csv data\demo\ground_truth.csv | Group-Object type | Select-Object Name, Count
```

What to explain:
* **Normal traffic:** a few very active users and many quiet ones (the busiest one at most
  20× the average). Each user logs in from
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
empty account: the account is in the summary with zeros.

Expected on `data\demo` (seed 42, W=60, N=5): 63 alarms (step 4 checks them against the labels).

Try other rule settings: `--threshold 3` or `--window 300`.

## 4. Detection quality ✅

```powershell
python -m login_correlation evaluate --data data\demo --results results\demo_ref
```
Expected output (1 M events, seed 42, W=60, N=5), measured on this laptop:
```
alarms            63  (true: 63, false: 0 = 0 on harmless scenarios + 0 from normal traffic)
attack scenarios  63 of 73 found
precision         1.0
recall            0.863
f1                0.9265
detection delay   mean 20.7 s, median 14.0 s, max 59 s (from the first event of the attack to the alarm, in log time)

scenario type         expected  scenarios  with alarm
brute_force              alarm         40          40
edge_exact_window         none         10           0
edge_inside_window       alarm         10          10
empty_username           alarm          3           3
same_timestamp           alarm         10          10
slow_brute               alarm         10           0
typo_burst                none         30           0
```
The full numbers are also saved in `results\demo_ref\metrics.json`.

What to explain:
* **Precision** = how many alarms are real attacks. **Recall** = how many attacks were found.
* The **edge cases behave exactly as specified**: 59 s span → alarm, 60 s span → no alarm,
  same second → alarm, empty user field → alarm for `<empty>`.
* **Recall is not 1.0 on purpose:** `slow_brute` attacks stay below N failures per W seconds,
  so this rule cannot see them. This is an honest limit of a single sliding-window rule.
* **Detection delay** is the log time from the attack's first event to the alarm. It is
  never more than W (59 s max), because the N-th failure must arrive first.
* On 10 M events (`data\d10m`) the result is almost the same: precision 0.98 (1 false alarm
  from a normal user's typos), recall 0.863.
* Change the rule and run `detect` + `evaluate` again. For example, `--threshold 3` makes
  the rule more sensitive: it can catch slower attacks but raises more false alarms.

## 5. OpenMP ✅

**a) Run it and check that the result is the same as the reference**
```powershell
python -m login_correlation detect --backend reference --data data\d10m --out results\d10m_ref
python -m login_correlation detect --backend openmp --threads 12 --split users --data data\d10m --out results\d10m_omp_users
python -m login_correlation detect --backend openmp --threads 12 --split time  --data data\d10m --out results\d10m_omp_time
python -m login_correlation compare results\d10m_ref results\d10m_omp_users
python -m login_correlation compare results\d10m_ref results\d10m_omp_time
Get-Content results\d10m_omp_time\timings.json
```
Expected: `IDENTICAL` for both files, both times. In `timings.json`, point out
`"threading_layer": "omp"` (real OpenMP threads), `"threads": 12`, `"chunks": 12` and
`"halo_events"`: the failures that a chunk had to read from before its own start, so that
windows crossing a chunk border are counted correctly.

**b) Speed-up**
```powershell
python -m login_correlation bench --data data\d10m      # about 1 min; 5 runs per setting
start results\plots\speedup_openmp.png
```
Expected (10 M events; numbers vary between runs - plug in the charger and close
browsers / Teams first):

| Threads | split by accounts | split by time + halo |
|---|---|---|
| 12, on battery | 0.14 s (2.0×) | 0.20 s (2.4×) |
| 12, on the charger | ≈ 0.12 s (≈ 3.8×) | ≈ 0.16 s (≈ 4×) |

What to explain:
* **Two ways to split the work**, as the task asks: by accounts (each thread gets whole
  users) and by time (each thread gets a time slice plus a halo of W seconds from the
  previous slice). Both give the same alarms as the reference.
* **Why not 12×?** The counting part does scale about 10×, but most of the remaining time is
  preparing data (limited by memory speed) and building tables (sequential). This is
  Amdahl's law. The laptop's 8 E-cores are also slower than its 2 P-cores.
* **In perspective:** detection on 10 M events takes 0.14 s, while reading the CSV and
  encoding it takes about 13 s.
* The design story (4 versions of the preparation step, see docs/ARCHITECTURE.md §7.1)
  shows why cache size and memory access patterns matter in parallel code.

## 6. MPI ✅

Open a terminal that was started **after** MS-MPI was installed (otherwise `mpiexec` is not found).

**a) Same result with real processes**
```powershell
mpiexec -n 4 python -m login_correlation detect --backend mpi --data data\demo --out results\demo_mpi4
python -m login_correlation compare results\demo_ref results\demo_mpi4
Get-Content results\demo_mpi4\timings.json
```
Expected: `IDENTICAL` twice, and a single `[mpi] ...` line (only process 0 prints and writes
files). In `timings.json` point out:
* `"ranks": 4`, `"events_per_rank"` (about 250 000 each) and `"accounts_per_rank"` (about 2 500
  each): every account - and so every alarm - belongs to exactly one process;
* the phases `partition`, `scatter`, `compute`, `gather`, and `local_min` / `local_max`
  (how evenly the work was split).

**b) Scaling** (about 5 minutes: every run reloads the 10 M-event CSV)
```powershell
python -m login_correlation bench --data data\d10m --what mpi
start results\plots\speedup_mpi.png
```
Expected: the local computation gets faster with more processes (≈ 2.9× on 8), but the whole
detection does not (≈ 0.5–0.8 s for every process count).

What to explain:
* **Partitioning by key:** `owner = crc32(user) % P`. All events of an account go to one process,
  so no process needs another's data and no alarm can be reported twice. `crc32` is used
  because Python's `hash()` gives different values in different processes.
* **Why MPI is not faster here:** process 0 must split the data, send it and collect the
  results (≈ 0.5 s), which costs more than the computation itself. MPI moves data between
  processes; OpenMP threads just share it. On a cluster each node would read its own part.
* The chart's left panel shows exactly this: the blue (compute) part shrinks, the grey
  (partition) and green (gather) parts do not.

## 7. OpenCL (GPU) ✅

**a) Same result on the graphics card**
```powershell
python -m login_correlation detect --backend opencl --device 0:0 --data data\demo --out results\demo_gpu
python -m login_correlation compare results\demo_ref results\demo_gpu
Get-Content results\demo_gpu\timings.json
```
Expected: `IDENTICAL` twice. In `timings.json` point out `"device": "Intel(R) Iris(R) Xe Graphics"`,
`"failures"` and one time per phase: `gpu_prepare`, `h2d`, `filter`, `sort`, `count`, `stats`,
`d2h`, `merge`, plus `encode` at the top (text → numbers, done before the GPU part).

**b) Where the time goes** (about 1 minute)
```powershell
python -m login_correlation bench --data data\d10m --what opencl
start results\plots\opencl_phases.png
```
Expected (10 M events): load ≈ 4-8 s, **encode ≈ 6-12 s**, the whole GPU pipeline ≈ 0.2-0.45 s,
of which the window count is only ≈ 3-5 ms.

What to explain:
* The task's pipeline step by step: **encode** the text fields (user name → number), then on
  the GPU **filter** the failures (`copy_if`), **sort** them (radix sort on one 64-bit key
  `user << 32 | time`, which is the same as sorting by user and then by time) and **count**
  (one GPU thread per failure looks back with a binary search; per-user counts use atomic
  operations).
* **Encoding is the expensive part** - many times slower than everything the GPU does. That
  is why the task asks to show it separately.
* The laptop's GPU shares memory with the CPU, so copying data to it is cheap; on a separate
  graphics card the copies would matter more.

## 8. Same results everywhere ✅
Each of steps 5-7 ends with `compare` against the reference, and `pytest` checks every backend
on random data, generated data and all edge cases (309 tests, none skipped).

## 9–10. Coming with the next milestones ⬜
These sections are filled in as each milestone is finished (see [ROADMAP.md](ROADMAP.md)).
