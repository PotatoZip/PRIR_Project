# Roadmap

### M0: Setup ✅

- [x] Project structure (docs, src, tests, config, data), `pyproject.toml`, `.venv`
- [x] Settings in `config/default.toml`, small sample dataset in `data/sample/`
- [x] Exact rule definition (docs/ARCHITECTURE.md §2) and `backends/reference.py`
- [x] Edge-case tests (window edge, repeated timestamps, empty accounts, empty user field, merging)
- [x] `python -m login_correlation check` reports OpenMP, MPI (MS-MPI) and OpenCL (Iris Xe) as working

### M1: Data generator (`generator.py`) ✅

- [x] Normal traffic: active and quiet users, random times, typos
- [x] All scenarios from docs/ARCHITECTURE.md §8.1 (including the edge cases and `empty_username`), saved in `ground_truth.csv`
- [x] Accounts with no events written to `users.csv`
- [x] The same `--seed` gives the same files
- [x] Tests: `tests/test_generator.py` (20 tests: format, repeatability, every scenario behaves as labelled)
- [x] Demonstration steps written in `docs/DEMO.md` §2–3
- **Done when:** 10 M events are generated in under 2 min (1 M in under 30 s), and the reference finds every scenario that should alarm except `slow_brute`.
- **Result (2026-11-09):** 1 M events in about 4 s; 10 M (with `--users 100000`) in about 1 min (400 MB).
  All expected scenarios except `slow_brute` are found; no alarm for `typo_burst` or
  `edge_exact_window`.
- **Fix made during M2:** user activity is capped at 20× the average. Without the cap the busiest
  account logged in every few seconds for a whole week, and its ordinary typos produced
  hundreds of false alarms on 10 M events.

### M2: Fast core + quality (`core.py`, `evaluate.py`) ✅

- [x] `core.py`: `sort_failures`, `window_counts` (binary search, `prange`), `find_episodes`, `build_summary`, `detect`, `warm_up`
- [x] `evaluate.py`: precision, recall, F1, results per scenario type, detection delay, false positives split by cause
- [x] `python -m login_correlation evaluate` prints a readable report and saves `metrics.json`
- [x] Faster encoding (`common.encode`): text → numbers on 10 M events went from 9.9 s to 5.5 s
- [x] One output format for all backends: `common.normalize_*` also fixes the column types
- [x] Tests: `tests/test_core.py` (core = reference on random and generated data, building blocks),
  `tests/test_evaluate.py` (metrics on hand-made cases), and every edge-case test in
  `tests/test_rule.py` now runs on both `reference` and `core`
- [x] Demonstration step written in `docs/DEMO.md` §4
- **Done when:** the core with 1 thread gives the same output as `reference`, and `python -m login_correlation evaluate` prints the metrics.
- **Result (2026-11-09):** core output is identical to reference in every test.

  | Data | core (prepare + compute + merge) | reference | Precision | Recall |
  |---|---|---|---|---|
  | 1 M events, 10 k users | 0.09 s | 1.6 s | 1.00 | 0.863 |
  | 10 M events, 100 k users | 0.39 s | 16 s | 0.98 | 0.863 |

  Recall is 0.863 because the 10 `slow_brute` attacks are missed, as designed (63 of 73 found).
  On 10 M events, reading the CSV (about 5 s) and encoding (about 5.5 s) now take far longer
  than the detection itself. This is the Amdahl's law point for the report.

### M3: OpenMP (`backends/openmp.py`) ✅

- [x] Variant A `--split users`: parallel grouping by user → `prange` over users; `--threads`
- [x] Variant B `--split time`: time chunks per thread + halo of W s (docs/ARCHITECTURE.md §7.1)
- [x] Parallel preparation (`group_by_user`, radix partitioning) and parallel per-user statistics
- [x] JIT warm-up, not included in the timings (`compile` is reported separately; later runs
  load the compiled code from Numba's disk cache in under 1 s)
- [x] Load balancing: users mixed by size (`balance=True`, default) vs. not mixed (tested both)
- [x] `python -m login_correlation bench --data data\d10m` → `results/bench_openmp.csv` + `results/plots/speedup_openmp.png`
- [x] Tests: `tests/test_openmp.py` (46 tests: building blocks, chunk border inside an attack,
  all failures in one second, more threads than failures, both splits × 1–12 threads,
  benchmark code) + 8 OpenMP settings in `tests/test_backends.py`
- [x] Demonstration step written in `docs/DEMO.md` §5
- **Done when:** both variants give output identical to `reference` for every thread count
  (including the test where a chunk border cuts through an attack), and you have a speed-up
  chart comparing A and B.
- **Result (2026-11-09):** output identical to `reference` in all tests, and byte-identical
  CSV files on 1 M and 10 M events for both splits. 10 M events, median of 5 runs
  (laptop **on battery**; repeat on the charger for the report):

  | Threads | split by accounts | speed-up | split by time + halo | speed-up |
  |---|---|---|---|---|
  | 1 | 0.285 s | 1.00 | 0.480 s | 1.00 |
  | 2 | 0.262 s | 1.09 | 0.353 s | 1.36 |
  | 4 | 0.228 s | 1.25 | 0.308 s | 1.56 |
  | 6 | 0.175 s | 1.63 | 0.246 s | 1.95 |
  | 8 | 0.147 s | 1.94 | 0.222 s | 2.16 |
  | 12 | 0.139 s | 2.05 | 0.197 s | 2.43 |

  The counting kernel itself scales well (time split: 0.140 s → 0.014 s, 10× on 12 threads),
  but it is a small part of the work. Preparing the data is limited by memory speed, and
  building the output tables is sequential (Amdahl's law). The 2 P-cores + 8 E-cores also
  limit the gain after 4 threads. Detection takes 0.14 s, while reading and encoding the
  CSV take about 13 s, so the parallel part is already not the bottleneck.
- **Re-measured on the charger (2026-11-17):** 12 threads take about 0.11–0.16 s, a speed-up of
  about **3.8–4.1×** for both splits. The runs were noisy because Firefox, Teams and Edge were
  busy in the background; for the report, close them and run `bench --what openmp` again.
- **Lessons (four versions of the preparation step):** per-thread arrays were too big for the
  cache (slower with more threads); a full scatter was limited by memory speed; "owner
  computes" suffered from branch mispredictions; **radix partitioning** finally scaled.
  The details are in the `openmp.py` docstring, which is good material for the report.

### M4: MPI (`backends/mpi.py`) ✅

- [x] `owner = crc32(user) % P` (the same in every process, unlike Python's `hash()`)
- [x] Rank 0: `partition()` - one-pass counting sort (Numba) of the events into one block per
  rank, keeping file order; then `Scatterv` of the NumPy arrays; each rank runs `core.detect`
  on its own accounts; `gather` + `merge()` on rank 0
- [x] Phases timed separately with `MPI.Wtime`: partition, scatter, compute (slowest rank),
  gather; plus `local_min` / `local_max` (load balance) and events / accounts per rank
- [x] `--threads` = Numba threads per process (default 1 = pure MPI)
- [x] `python -m login_correlation bench --data data\d10m --what mpi` → `results/bench_mpi.csv`
  + `results/plots/speedup_mpi.png`
- [x] Tests: `tests/test_mpi.py` (42 tests: ownership, partition, simulated ranks = reference for
  1–16 ranks, more ranks than accounts, real `mpiexec` runs with 2/3/5/8 processes on generated
  data, other rule settings, only rank 0 writes, benchmark code) + `test_mpi_matches_reference`
  (1–4 processes) and MPI in `tests/test_backends.py` - all pass (267 tests in total)
- [x] Demonstration step written in `docs/DEMO.md` §6
- [ ] _Optional (only if there is time):_ time chunks with overlap
- **Done when:** the `test_mpi_matches_reference` test passes for 1–4 processes, and you have a scaling chart.
- **Result (2026-11-17):** byte-identical CSV files on 1 M and 10 M events for 1, 2, 4 and 8
  processes; precision and recall unchanged. 10 M events, median of 3 runs, on the charger:

  | Processes | partition | scatter | compute (slowest rank) | gather | whole detection | speed-up |
  |---|---|---|---|---|---|---|
  | 1 | 0.12 s | 0.06 s | 0.270 s | 0.06 s | 0.50 s | 1.00 |
  | 2 | 0.35 s | 0.09 s | 0.242 s | 0.09 s | 0.78 s | 0.64 |
  | 4 | 0.32 s | 0.07 s | 0.139 s | 0.09 s | 0.62 s | 0.80 |
  | 6 | 0.24 s | 0.06 s | 0.114 s | 0.11 s | 0.62 s | 0.81 |
  | 8 | 0.31 s | 0.07 s | 0.094 s | 0.12 s | 0.63 s | 0.79 |

  **The local computation scales (2.9× on 8 processes), but the whole detection does not get
  faster on one laptop.** Rank 0 alone partitions the data, sends it and collects the results
  (≈ 0.5 s), which is more than the whole computation on one process (0.27 s) - Amdahl's law
  plus communication cost. Partitioning also gets slower when more processes run (0.12 s →
  ~0.3 s), most likely because MS-MPI processes that wait for data keep polling and take CPU
  time and memory bandwidth from rank 0. On a real cluster each node would read its own part
  of the data, which removes most of this overhead. This is the main MPI lesson for the report.
- **Fix made during M4:** partitioning first used `np.argsort` + three copies of 10 M-element
  arrays (1.3 s on 8 ranks); a one-pass counting sort in Numba takes ~0.1–0.3 s.

### M5: OpenCL (`backends/opencl.py`) ✅

- [x] Device choice: `--device P:D` or `PYOPENCL_CTX`; default = first GPU of a native platform
  (Microsoft's `OpenCLOn12` layer is skipped)
- [x] Copy to the GPU, key `user << 32 | ts_rel` (ElementwiseKernel), `copy_if` filter, `RadixSort`
  (only the used key bits: 49 for 100 000 users), `window_count` kernel (binary search per
  failure + `atomic_max` per user), `user_stats` kernel (`atomic_inc` / `atomic_min` / `atomic_max`)
- [x] Every phase timed separately (wall clock after `queue.finish()`); encoding text → numbers is
  measured on its own by the command line; kernel build + first run reported as `compile`
- [x] Alarms built on the CPU with the same `core.find_episodes` as every other backend
- [x] `python -m login_correlation bench --data data\d10m --what opencl` → `results/bench_opencl.csv`
  + `results/plots/opencl_phases.png`
- [x] New dependency: `mako` (PyOpenCL needs it for `RadixSort` / `copy_if`), added to `requirements.txt`
- [x] Tests: `tests/test_opencl.py` (33 tests: device choice, each GPU kernel against the CPU code,
  GPU radix sort against numpy, whole backend = reference on random data and 5 rule settings,
  window edges, equal timestamps, empty user field, no failures, no events, real epoch
  timestamps, too long time span, phase timings, command line, benchmark code) + OpenCL in
  `tests/test_backends.py`. **All 309 tests pass; nothing is skipped any more.**
- [x] Demonstration step written in `docs/DEMO.md` §7
- **Done when:** the output equals `reference`, and you have a phase-breakdown chart.
- **Result (2026-11-17):** byte-identical CSV files on 1 M and 10 M events; precision and
  recall unchanged. 10 M events (432 336 failures), Intel Iris Xe, median of 5 runs, on the
  charger (noisy: browsers and Teams were running):

  | Phase | Where | Median time |
  |---|---|---|
  | load CSV | CPU | 4.4 s |
  | **encode** (text → numbers) | CPU | **6.4 s** |
  | gpu_prepare (ts → uint32 offset) | CPU | 111 ms |
  | copy to GPU (h2d) | | 40 ms |
  | filter (`copy_if`) | GPU | 68 ms |
  | radix sort | GPU | 120 ms |
  | window count | GPU | 4.5 ms |
  | per-user stats (atomics) | GPU | 16 ms |
  | copy back (d2h) | | 17 ms |
  | alarms + tables (merge) | CPU | 92 ms |
  | **whole GPU pipeline** | | **0.44 s** (0.22 s in a quieter run) |

  **Encoding the text costs about 15-30× more than the whole GPU pipeline**, and the window
  count itself takes only a few milliseconds. Sorting is the biggest GPU phase. The Iris Xe
  shares memory with the CPU, so copies are cheap (40 ms for 90 MB); on a separate graphics
  card they would cost more.

### M6: Stream replay (`stream.py`) ✅

- [x] `stream.replay`: events replayed in (time, file order) with `--speedup` (log seconds per
  real second, `inf` = as fast as possible); a deque of the last W seconds of failures per
  account; an alarm is raised the moment the N-th failure arrives; overlapping triggers
  extend the open alarm, exactly as in batch mode
- [x] Measured per alarm: processing latency (N-th failure arrives → alarm), time to alarm in
  real time (first failure of the window arrives → alarm), detection delay in log time, and
  the max lag (whether the detector keeps up); p50 / p95 / max
- [x] `python -m login_correlation replay --data … --speedup … [--out …]` prints the report and
  checks that the alarms equal batch detection (`same alarms as batch detection: YES`);
  `--out` saves `alarms.csv`, `per_alarm.csv`, `replay.json`
- [x] `compare` now also accepts a replay folder (it has no `summary.csv`, so only the alarms are compared)
- [x] `python -m login_correlation bench --data data\demo --what stream` → `results/bench_stream.csv`
  + `results/plots/stream_latency.png` (speed-ups from `[bench].stream_speedups`)
- [x] Tests: `tests/test_stream.py` (38 tests: replay = batch on random data and 5 rule settings,
  every edge case, file not in time order, alarm raised at the N-th failure, one raised alarm
  per final alarm, paced replay follows the clock, invalid speed-up, command line, `compare`
  with replay output, benchmark code)
- [x] Demonstration step written in `docs/DEMO.md` §9
- **Done when:** you have latency results for 3 speed-up values.
- **Result (2026-11-17):** replayed alarms are identical to batch detection on 1 M and 10 M
  events. 1 M events (63 alarms), Python loop, laptop on the charger:

  | Replay speed-up | Replay took | Time to alarm (real) p50 / p95 / max | Processing p50 / max | Max lag |
  |---|---|---|---|---|
  | ∞ (as fast as possible) | 0.18 s | 0.007 / 0.017 / 0.021 ms | 1.1 / 6.9 µs | - |
  | 100 000× | 6.0 s | 0.14 / 0.60 / 0.60 ms | 2.8 / 6.0 µs | 12 ms |
  | 10 000× | 60.5 s | 1.4 / 5.9 / 5.9 ms | 3.8 / 10.6 µs | 18 ms |

  The detector itself needs only microseconds per alarm; the real-time "time to alarm" is
  the rule's detection delay (median 14 s, max 59 s of log time, because the N-th failure has
  to arrive) divided by the speed-up. In real time (1×) an alarm comes about 14 s after the
  attack starts, plus a few microseconds. Unpaced, the replay handles about 2-5 M events/s
  (10 M events in 5.6 s), far more than any real login stream.

### M7: Benchmarks, report, presentation (`bench.py`)

- [ ] Run the experiments (docs/ARCHITECTURE.md §8.4) → `results/bench.csv` + PNG charts
- [ ] Write the report (outline below)
- [ ] Test `demo.ps1` on the laptop; keep the charts ready as a backup
- **Done when:** the report and slides are ready, and `demo.ps1` runs in under 3 minutes.
