cmake_minimum_required(VERSION 3.20)
project(ISM CXX)

set(CMAKE_CXX_STANDARD 20)

set(WARNINGS "-Wall -Wextra -Wno-assume -Wno-unused-command-line-argument -Wno-tautological-constant-compare")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -fomit-frame-pointer -march=native -mtune=native ${WARNINGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined -fPIE ${WARNINGS}")

set(DPCPP_FLAGS -fsycl-targets=nvptx64-nvidia-cuda,spir64 -fsycl-id-queries-fit-in-int -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_75 -Xcuda-ptxas -v -fgpu-inline-threshold=10000)
# -fcuda-approx-transcendentals -ffp-contract=fast -fcuda-flush-denormals-to-zero -Xsycl-target-backend=spir64_x86_64 --march=avx
#set(DPCPP_FLAGS -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx902 -fsycl-id-queries-fit-in-int)
#set(DPCPP_FLAGS -fsycl-targets=spir64 -fsycl-id-queries-fit-in-int)

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

# Sequential library
add_library(ISM_SEQ_backend SHARED src/sim_seq.cpp include/internal/utils.hpp include/sim_seq.h)
add_sycl_to_target(TARGET ISM_SEQ_backend SOURCES src/sim_seq.cpp)

add_executable(ISM_SEQ main_seq.cpp)
add_sycl_to_target(TARGET ISM_SEQ SOURCES main_seq.cpp)
target_link_libraries(ISM_SEQ PRIVATE ISM_SEQ_backend)

# SYCL Library
add_library(ISM_SYCL_backend SHARED src/sim_sycl.cpp include/internal/utils.hpp include/sim_sycl.h)
add_sycl_to_target(TARGET ISM_SYCL_backend SOURCES src/sim_sycl.cpp)

add_executable(ISM_SYCL main_sycl.cpp)
add_sycl_to_target(TARGET ISM_SYCL SOURCES main_sycl.cpp)
target_link_libraries(ISM_SYCL PRIVATE ISM_SYCL_backend)


include(FetchContent)

if (BUILD_TESTING)
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
