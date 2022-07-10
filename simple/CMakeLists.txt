cmake_minimum_required(VERSION 3.10)

project(simple)

add_definitions(-std=c++11 -O2 -pthread)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
include_directories(${PARENT_DIR}/neural-network/include/common)
include_directories(${PARENT_DIR}/neural-network/include/logger)
include_directories(${PARENT_DIR}/neural-network/include/tensorrt-base)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda/targets/x86_64-linux/include)
link_directories(/usr/local/cuda/targets/x86_64-linux/lib)

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_61;code=sm_61;-gencode;arch=compute_60;code=sm_60;-gencode;arch=compute_62;code=sm_62)

file(GLOB_RECURSE Project_Sources
        ${PROJECT_SOURCE_DIR}/include/*
        ${PROJECT_SOURCE_DIR}/src/*
        ${PARENT_DIR}/neural-network/include/common/*
        ${PARENT_DIR}/neural-network/src/common/*
        ${PARENT_DIR}/neural-network/include/logger/*
        ${PARENT_DIR}/neural-network/include/tensorrt-base/*
        ${PARENT_DIR}/neural-network/src/tensorrt-base/*)

cuda_add_executable(${PROJECT_NAME} ${Project_Sources} )
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})


