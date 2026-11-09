# PRIR_Project: Korelacja zdarzeń logowania w logach

## Project structure

```
docs/        ARCHITECTURE.md (how it works and why), ROADMAP.md (plan), Opis_Tematu.docx (topic)
src/         the program: src/login_correlation/ (backends/ holds the 4 versions)
tests/       pytest tests
config/      default.toml - rule settings (W, N), generator and benchmark settings
data/        sample/ dataset (in git) + generated datasets (not in git)
results/     outputs of runs (created automatically, not in git)
demo.ps1     runs everything for the presentation
```

The full file tree with a description of every file is in [docs/ARCHITECTURE.md §4](docs/ARCHITECTURE.md#4-project-structure).

## Setup (Windows 11)

1. **Python and the packages**, in PowerShell inside the project folder:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   pip install -e .                  # installs requirements.txt + this project
   ```
2. **OpenMP**: nothing to install; Numba includes it.
3. **MPI**: install `msmpisetup.exe` (MS-MPI v10.1.3) from Microsoft's
   [MS-MPI page](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi),
   then open a **new** terminal.
4. **OpenCL**: comes with the Intel Graphics driver. If it is missing, update the driver
   with _Intel Driver & Support Assistant_. Use device `0:0`.
5. **Check:** `python -m login_correlation check` should end with `All good.`

## How to run

```powershell
python -m login_correlation check                                    # are OpenMP, MPI, OpenCL working?
pytest                                                               # run all tests
python -m login_correlation detect --data data\sample --out results\sample   # try it on the sample data

python -m login_correlation generate --out data\demo --events 1000000          # create test data (M1)
python -m login_correlation detect --backend reference --data data\demo --out results\ref
python -m login_correlation detect --backend openmp --threads 8 --data data\demo --out results\omp
python -m login_correlation detect --backend openmp --threads 8 --split time --data data\demo --out results\omp_time
mpiexec -n 4 python -m login_correlation detect --backend mpi --data data\demo --out results\mpi
python -m login_correlation detect --backend opencl --data data\demo --out results\gpu
python -m login_correlation compare results\ref results\omp          # must be IDENTICAL
python -m login_correlation evaluate --data data\demo --results results\ref    # precision / recall
python -m login_correlation replay --data data\demo --speedup 1000   # time-to-alarm
.\demo.ps1                                                           # all of the above
```

Settings come from `config/default.toml`. Command-line options override them, e.g.
`--window 30 --threshold 10`. Commands that are not written yet print "not implemented yet"
together with their milestone.
