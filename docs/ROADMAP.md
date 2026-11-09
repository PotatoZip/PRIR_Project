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
- **Result (2026-11-09):** 1 M events in 5.1 s; 10 M in 54 s (403 MB). On 1 M events the reference
  raises 65 alarms. 63 match labelled attacks; all expected scenarios except `slow_brute` are
  found; no alarm for `typo_burst` or `edge_exact_window`. 2 false positives come from very
  active normal users.

### M2: Fast core + quality (`core.py`, `evaluate.evaluate`)

- [ ] `window_counts` with `lower_bound`, `merge_episodes`, `build_summary`
- [ ] `evaluate`: precision, recall, results per scenario type, detection delay
- **Done when:** the core with 1 thread gives the same output as `reference`, and `python -m login_correlation evaluate` prints the metrics.

### M3: OpenMP (`backends/openmp.py`)

- [ ] Variant A `--split users`: filter → sort → segments → `prange` over users; `--threads`
- [ ] Variant B `--split time`: time chunks per thread + halo of W s (docs/ARCHITECTURE.md §7.1)
- [ ] JIT warm-up, not included in the timings
- [ ] Load balancing: compare "no balancing" with "users mixed by size"
- **Done when:** both variants give output identical to `reference` for every thread count
  (including the test where a chunk border cuts through an attack), and you have a speed-up
  chart comparing A and B.

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
