"""The four versions of the detector. Each module has the same function:

    detect(events: EventArrays, params: RuleParams, **options) -> DetectionResult

reference.py  plain Python, the correct answer
openmp.py     Numba prange on OpenMP threads        (TODO M3)
mpi.py        mpi4py + MS-MPI processes             (TODO M4)
opencl.py     PyOpenCL on the GPU                   (TODO M5)

Pick one by name with common.get_backend("openmp").
"""
