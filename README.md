# Building instructions

## To build the SYCL version: 
```
mkdir -p build ; cd build
CXX=clang++ cmake .. -DCMAKE_BUILD_TYPE=Release
make ISM_SYCL
./ISM_SYCL ../particule.xyz
```
For DPC++/Clang, eventually change the target flags in [CMakeLists.txt](CMakeLists.txt) in `DPCPP_FLAGS` and select the device with the env variable `SYCL_DEVICE_FILTER=...`.

## Sequential/Reference version:
The target to build is `ISM_SEQ` and produces an executable of the same name.

## Benchmark
The target to build is `benchmark_lennard_jones`.
