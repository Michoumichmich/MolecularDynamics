# Molecular Dynamics

Implements Velocity Verlet using a Lennard Jones force field. Outputs PDB files. Supports domain decomposition on OpenMP. Offers a SYCL backend-too (but no domain decomposition there yet).

# Building instructions

### With hipSYCL

Use the supplied script to get a version of hipSYCL and build the simulator with it.

```
chmod u+x build.sh
./build.sh
```

### With Intel(R) oneAPI or LLVM:

Just use `clang++` (with SYCL support) or `dpcpp` as the default C++ compiler:

```
mkdir -p build ; cd build
CXX={clang++|dpcpp} cmake .. -DSYCL={CPU|HIP|CUDA}
make main
./main ../particule.xyz
```

For DPC++/Clang, eventually change the target flags in [CMakeLists.txt](CMakeLists.txt) in `DPCPP_FLAGS` and select the device with the env variable `SYCL_DEVICE_FILTER=...`.

## Sequential/Reference version:

Set `-DSYCL` to `OFF`.

## Benchmark

Pass `-DBUILD_BENCH=ON` to cmake and the target to build is `benchmark_lennard_jones` or `benchmark_decompose`.

# Execution

To run the simulation with domain decomposition run:

```
./main particule.xyz -decompose
```

Otherwise, to run it on SYCL (domain decomposition not supported yet), replace `decompose` by `sycl`. Omitting the flag will run the default simulation (C++) without domain decomposition.
