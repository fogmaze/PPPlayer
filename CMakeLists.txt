cmake_minimum_required(VERSION 3.0.0)
project(ball_simulate VERSION 0.1.0)

add_library(dataFileOperator SHARED ball_simulate/dataFileOperator.cpp)

set(CMAKE_PREFIX_PATH cpp_packages/libtorch)
include_directories(cpp_packages/libtorch/include)
include_directories(cpp_packages/libtorch/include/torch/csrc/api/include)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(dataFileOperator "${TORCH_LIBRARIES}")