#!/bin/sh
export MD_HOME=$PWD

mkdir -p build
cd build || exit

# get a version of hipSYCL
(if cd hipSYCL; then git pull; else git clone https://github.com/illuhad/hipSYCL.git; fi)
cd hipSYCL || exit
mkdir -p build
cd build || exit
cmake .. -DCMAKE_INSTALL_PREFIX=$MD_HOME/build/hipSYCL/install -DCMAKE_BUILD_TYPE=Release
make install -j $(nproc)
export hipSYCL_DIR=$MD_HOME/build/hipSYCL/install

# Finally build the project
cd $MD_HOME/build || exit
cmake .. -DSYCL=ON -DBUILD_FLOAT=ON -DBUILD_DOUBLE=ON -DBUILD_BENCH=ON -DHIPSYCL_TARGETS=omp && cmake --build . -t  main
