# PRIR_Project: Korelacja zdarzeń logowania w logach

This file covers **setup and configuration only**.
* Running and presenting the project: [docs/DEMO.md](docs/DEMO.md)
* How it works: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
* Plan and progress: [docs/ROADMAP.md](docs/ROADMAP.md)

## Project structure

```
docs/        ARCHITECTURE.md, ROADMAP.md, DEMO.md, Opis_Tematu.docx (topic)
src/         the program: src/login_correlation/ (backends/ holds the 4 detector versions)
tests/       pytest tests
config/      default.toml - all settings (see "Configuration" below)
data/        sample/ dataset (in git) + generated datasets (not in git)
results/     outputs of runs (created automatically, not in git)
demo.ps1     runs the whole demonstration
```
The full file tree with a description of every file is in [docs/ARCHITECTURE.md §4](docs/ARCHITECTURE.md#4-project-structure).

## Setup (Windows 11)

| Step | What to do | Needed for |
|---|---|---|
| 1 | Python 3.11 or newer ([python.org](https://www.python.org/downloads/)); tested with 3.12 | everything |
| 2 | Create the environment (commands below) | everything |
| 3 | Install **MS-MPI v10.1.3**: run `msmpisetup.exe` from Microsoft's [MS-MPI page](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi), then open a **new** terminal | MPI backend |
| 4 | Intel Graphics driver (includes OpenCL). If it is missing, update with *Intel Driver & Support Assistant* | OpenCL backend |
| — | OpenMP: nothing to install, Numba includes it | OpenMP backend |

Step 2, in PowerShell inside the project folder:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e .                  # installs requirements.txt + this project (src/)
```
Activate the environment (`.\.venv\Scripts\Activate.ps1`) in every new terminal.

## Check the setup

```powershell
python -m login_correlation check     # must end with "All good."
pytest                                # all tests; backends that are not written yet are "skipped"
```

| `check` line | If it says FAIL |
|---|---|
| `OpenMP (Numba)` | `pip install -e .` again |
| `mpiexec` / `mpi4py` | install MS-MPI (step 3) and open a new terminal |
| `OpenCL …` | update the Intel Graphics driver (step 4) |

## Configuration

All settings are in [config/default.toml](config/default.toml). A command-line option
overrides the file for one run. Use `--config other.toml` for a different file.

| Section | Key | Default | Meaning | Command-line option |
|---|---|---|---|---|
| `[rule]` | `window_s` | 60 | **W**: window length in seconds; the window is `(t − W, t]` | `--window` |
| | `threshold` | 5 | **N**: failed logins of one account in the window that raise an alarm | `--threshold` |
| `[generator]` | `seed` | 42 | random seed; the same seed gives identical files | `--seed` |
| | `n_users` | 10000 | number of accounts with traffic | `--users` |
| | `n_events` | 1000000 | approximate number of normal events | `--events` |
| | `start_ts` | 1767225600 | first timestamp (2026-01-01 00:00 UTC) | — |
| | `duration_s` | 604800 | time span of the data (7 days) | — |
| | `typo_rate` | 0.03 | share of normal logins with 1–2 typos first | — |
| | `n_empty_users` | 50 | accounts in `users.csv` that have no events | — |
| `[generator.scenarios]` | `<type> = count` | 3–40 | how many labelled scenarios of each type to insert (types: docs/ARCHITECTURE.md §8.1) | — |
| `[bench]` | `repeats`, `sizes`, `threads`, `ranks` | 5, … | benchmark settings (milestone M7) | — |

Backend options (command line only):

| Option | Backend | Meaning |
|---|---|---|
| `--backend` | all | `reference` (default), `openmp`, `mpi`, `opencl` |
| `--threads N` | openmp | number of OpenMP threads |
| `--split users\|time` | openmp | split work by accounts (default) or by time chunks with a halo |
| `--device P:D` | opencl | OpenCL platform:device; use `0:0` (native Intel driver) on this laptop |
| `mpiexec -n P …` | mpi | number of MPI processes |

Environment variables (optional):

| Variable | Default | Meaning |
|---|---|---|
| `NUMBA_THREADING_LAYER` | `omp` (set by the program) | keep it `omp` so Numba uses OpenMP |
| `NUMBA_NUM_THREADS` | number of logical cores (12) | upper limit for `--threads` |
| `PYOPENCL_CTX` | — | default OpenCL device, e.g. `0:0` |
