cmake_minimum_required(VERSION 3.0)
project(ISM CXX)

set(CMAKE_CXX_STANDARD 20)

set(WARNINGS "-Wall -Wextra -Wno-assume -Wno-unused-command-line-argument -Wno-tautological-constant-compare -Wno-unknown-warning-option -Wno-linker-warnings")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -fomit-frame-pointer -march=native -mtune=native ${WARNINGS} -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined -fPIE ${WARNINGS}")

if (APPLE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
endif ()


if (SYCL STREQUAL CUDA)
    set(DPCPP_FLAGS -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_75 -Wno-unknown-cuda-version -Xcuda-ptxas -v -fgpu-inline-threshold=10000)
elseif (SYCL STREQUAL HIP)
    set(DPCPP_FLAGS -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx902 -fsycl-id-queries-fit-in-int -fgpu-inline-threshold=1000)
elseif (SYCL STREQUAL CPU)
    set(DPCPP_FLAGS -fsycl-targets=spir64_x86_64)
endif ()

include(cmake/FindSYCL.cmake)

option(BUILD_DOUBLE "Build with double backend" ON)
option(BUILD_FLOAT "Build with float backend" ON)
option(BUILD_HALF "Build with half backend" OFF)

if (BUILD_DOUBLE)
    add_definitions(-DBUILD_DOUBLE)
endif ()
if (BUILD_FLOAT)
    add_definitions(-DBUILD_FLOAT)
endif ()
if (BUILD_HALF)
    add_definitions(-DBUILD_HALF)
endif ()


include_directories(include)

add_executable(main main.cpp)
add_sycl_to_target(TARGET main SOURCES main.cpp)


# CPU BE
add_library(cpu_backend SHARED src/cpu_backend.cpp)
add_sycl_to_target(TARGET cpu_backend SOURCES src/cpu_backend.cpp)
target_link_libraries(main PRIVATE cpu_backend)

if (SYCL)
    # SYCL Library
    add_library(sycl_backend SHARED src/sycl_backend.cpp)
    add_sycl_to_target(TARGET sycl_backend SOURCES src/sycl_backend.cpp)
    target_link_libraries(main PRIVATE sycl_backend)
endif ()


include(FetchContent)

if (BUILD_BENCH)
    # Externally provided libraries
    FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG main)

    FetchContent_Declare(googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG main) # need master for benchmark::benchmark

    FetchContent_MakeAvailable(
            googletest
            googlebenchmark)

    include(benchmarks/CMakeLists.txt)
endif ()
