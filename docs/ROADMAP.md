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
- **Lessons (four versions of the preparation step):** per-thread arrays were too big for the
  cache (slower with more threads); a full scatter was limited by memory speed; "owner
  computes" suffered from branch mispredictions; **radix partitioning** finally scaled.
  The details are in the `openmp.py` docstring, which is good material for the report.

### M4: MPI (`backends/mpi.py`)

- [ ] `owner = crc32(user) % P`, `Scatterv`, local computation, `gather` to rank 0
- [ ] Time scatter, compute and gather separately
- [ ] _Optional (only if there is time):_ time chunks with overlap
- **Done when:** the `test_mpi_matches_reference` test passes for 1–4 processes, and you have a scaling chart.

### M5: OpenCL (`backends/opencl.py`)

- [ ] Copy to the GPU, `copy_if`, `RadixSort` on the combined key, window-count kernel, per-user counts
- [ ] Time every phase (OpenCL profiling events), with encoding timed separately
- **Done when:** the output equals `reference`, and you have a phase-breakdown chart.

### M6: Stream replay (`evaluate.replay`)

- [ ] Replay in time order with `--speedup`; queue of recent failures per user
- [ ] Latency p50 / p95 / max; the alarms are the same as in batch mode
- **Done when:** you have latency results for 3 speed-up values.

### M7: Benchmarks, report, presentation (`bench.py`)

- [ ] Run the experiments (docs/ARCHITECTURE.md §8.4) → `results/bench.csv` + PNG charts
- [ ] Write the report (outline below)
- [ ] Test `demo.ps1` on the laptop; keep the charts ready as a backup
- **Done when:** the report and slides are ready, and `demo.ps1` runs in under 3 minutes.
