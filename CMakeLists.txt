find_package(PythonInterp REQUIRED)
cmake_minimum_required(VERSION 3.0.0)
project(ball_simulate VERSION 0.1.0)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")

IF (WIN32 AND NOT CYGWIN)
    SET(CMAKE_C_FLAGS_DEBUG "-D_DEBUG")
ENDIF ()

add_library(dataFileOperator SHARED ball_simulate/dataFileOperator.cpp)
add_executable(dataFileOperatorTest ball_simulate/dataFileOperator.cpp)
add_executable(physx_simulate ball_simulate/physx_simulate.cpp)

target_link_directories(dataFileOperatorTest PRIVATE cpp_packages/lib)
target_link_directories(dataFileOperator PRIVATE cpp_packages/lib)
target_link_directories(physx_simulate PRIVATE cpp_packages/lib)

include_directories(cpp_packages/include)

#set(CMAKE_PREFIX_PATH cpp_packages/libtorch)
#include_directories(cpp_packages/libtorch/include)
#include_directories(cpp_packages/libtorch/include/torch/csrc/api/include)

#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#target_link_libraries(dataFileOperator "${TORCH_LIBRARIES}")
#target_link_libraries(dataFileOperatorTest "${TORCH_LIBRARIES}")
# Set the directory where your .so and .a files are located

set(LIB_DIR "cpp_packages/lib")

# Generate a list of all .so and .a files in the specified directory
file(GLOB LIB_FILES "${LIB_DIR}/*.so" "${LIB_DIR}/*.a")
message(${LIB_FILES})

# Iterate over the list of library files and link them to your targe
foreach(LIB_FILE ${LIB_FILES})
    #message(${LIB_FILE})
    target_link_libraries(physx_simulate ${LIB_FILE})
endforeach()
