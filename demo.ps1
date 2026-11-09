# Runs the whole project for the presentation.   .\demo.ps1   or   .\demo.ps1 -Events 200000
# Steps that are not written yet print "not implemented yet" and the demo continues.

param([int]$Events = 1000000, [int]$Window = 60, [int]$Threshold = 5,
      [int]$Threads = 8, [int]$Ranks = 4)

Set-Location $PSScriptRoot
if (Test-Path ".venv\Scripts\Activate.ps1") { . .\.venv\Scripts\Activate.ps1 }
$data = "data\demo"; $res = "results\demo"
$rule = @("--window", $Window, "--threshold", $Threshold)
function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }

Step "0. Environment";                 python -m login_correlation check
Step "1. Generate $Events events";     python -m login_correlation generate --out $data --events $Events @rule
Step "2. Reference";                   python -m login_correlation detect --backend reference --data $data --out "$res\reference" @rule
Step "3. OpenMP, $Threads threads";    python -m login_correlation detect --backend openmp --threads $Threads --data $data --out "$res\openmp" @rule
Step "3b. OpenMP, time split + halo";  python -m login_correlation detect --backend openmp --threads $Threads --split time --data $data --out "$res\openmp_time" @rule
Step "4. MPI, $Ranks processes";       mpiexec -n $Ranks python -m login_correlation detect --backend mpi --data $data --out "$res\mpi" @rule
Step "5. OpenCL (GPU)";                python -m login_correlation detect --backend opencl --device 0:0 --data $data --out "$res\opencl" @rule

Step "6. Are all results identical?"
foreach ($b in "openmp", "openmp_time", "mpi", "opencl") {
    if (Test-Path "$res\$b\alarms.csv") { Write-Host "reference vs $b"; python -m login_correlation compare "$res\reference" "$res\$b" }
}

Step "7. Precision / recall";          python -m login_correlation evaluate --data $data --results "$res\reference"
Step "8. Time-to-alarm";               python -m login_correlation replay --data $data --speedup 1000 @rule
Write-Host "`nDone. Results in $res" -ForegroundColor Green
