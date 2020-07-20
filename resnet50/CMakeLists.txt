cmake_minimum_required(VERSION 3.8)

project(resnet50)

add_definitions(-std=c++11 -O2 -pthread)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda/targets/x86_64-linux/include)
link_directories(/usr/local/cuda/targets/x86_64-linux/lib)

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_61;code=sm_61;-gencode;arch=compute_60;code=sm_60;-gencode;arch=compute_62;code=sm_62)

file(GLOB_RECURSE resnet50_Sources
        ${PROJECT_SOURCE_DIR}/include/*
        ${PROJECT_SOURCE_DIR}/src/*)

cuda_add_executable(resnet50 ${resnet50_Sources} )
target_link_libraries(resnet50 nvinfer)
target_link_libraries(resnet50 cudart)
target_link_libraries(resnet50 ${OpenCV_LIBRARIES})


