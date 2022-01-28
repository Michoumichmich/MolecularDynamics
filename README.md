# Building instructions

## To build the SYCL backend:

```
mkdir -p build ; cd build
CXX=clang++ cmake .. -DCMAKE_BUILD_TYPE=Release -DSYCL=ON
make main
./main ../particule.xyz
```

For DPC++/Clang, eventually change the target flags in [CMakeLists.txt](CMakeLists.txt) in `DPCPP_FLAGS` and select the device with the env variable `SYCL_DEVICE_FILTER=...`.

## Sequential/Reference version:

Set -DSYCL to OFF.

## Benchmark

Pass `-DBUILD_BENCH=ON` to cmake and the target to build is `benchmark_lennard_jones`.
