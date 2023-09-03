set(CMAKE_CURRENT_BINARY_DIR build)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

find_package(PythonInterp REQUIRED)
cmake_minimum_required(VERSION 3.0.0)
project(ball_simulate VERSION 0.1.0)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")

IF (WIN32 AND NOT CYGWIN)
    SET(CMAKE_C_FLAGS_DEBUG "-D_DEBUG")
ENDIF ()

include_directories(cpp_packages/include)
set(LIB_DIR "cpp_packages/lib")

add_library(dataFileOperator SHARED ball_simulate/dataFileOperator.cpp)

target_link_directories(dataFileOperator PRIVATE cpp_packages/lib)

# dfo v2
add_library(dataFileOperatorV2-100-250 SHARED ball_simulate_v2/dataFileOperator.cpp)
add_library(dataFileOperatorV2-100-100 SHARED ball_simulate_v2/dataFileOperator.cpp)
add_library(dataFileOperatorV2-40-250 SHARED ball_simulate_v2/dataFileOperator.cpp)
add_library(dataFileOperatorV2-80-50 SHARED ball_simulate_v2/dataFileOperator.cpp)
add_library(dataFileOperatorV2-40-50 SHARED ball_simulate_v2/dataFileOperator.cpp)
target_compile_definitions(dataFileOperatorV2-100-250 PRIVATE -DINPUT_LEN=100 -DTEST_LEN=250)
target_compile_definitions(dataFileOperatorV2-100-100 PRIVATE -DINPUT_LEN=100 -DTEST_LEN=100)
target_compile_definitions(dataFileOperatorV2-40-250 PRIVATE -DINPUT_LEN=40 -DTEST_LEN=250)
target_compile_definitions(dataFileOperatorV2-80-50 PRIVATE -DINPUT_LEN=40 -DTEST_LEN=50)
target_compile_definitions(dataFileOperatorV2-40-50 PRIVATE -DINPUT_LEN=40 -DTEST_LEN=50)



#set(CMAKE_PREFIX_PATH cpp_packages/libtorch)
#include_directories(cpp_packages/libtorch/include)
#include_directories(cpp_packages/libtorch/include/torch/csrc/api/include)

#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#target_link_libraries(dataFileOperator "${TORCH_LIBRARIES}")
#target_link_libraries(dataFileOperatorTest "${TORCH_LIBRARIES}")
# Set the directory where your .so and .a files are located


# physx

#add_executable(physx_simulate ball_simulate/physx_simulate.cpp)
#target_link_directories(physx_simulate PRIVATE cpp_packages/lib)
#include_directories(cpp_packages/include/physx)

#set(PhysXLIBS
  #libPhysXExtensions_static_64.a
  #libPhysX_static_64.a
  #libPhysXPvdSDK_static_64.a
  #libPhysXVehicle_static_64.a
  #libPhysXVehicle2_static_64.a
  #libPhysXCharacterKinematic_static_64.a
  #libPhysXCooking_static_64.a
  #libPhysXCommon_static_64.a
  #libPhysXFoundation_static_64.a
  #libSnippetUtils_static_64.a
  #libSnippetRender_static_64.a
#)

#target_link_libraries(physx_simulate PRIVATE ${PhysXLIBS})
