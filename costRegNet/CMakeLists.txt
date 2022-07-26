cmake_minimum_required(VERSION 3.10)

project(cost_reg_net)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
include_directories(${PARENT_DIR}/neural-network/include/common)
include_directories(${PARENT_DIR}/neural-network/include/logger)
include_directories(${PARENT_DIR}/neural-network/include/tensorrt-base)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda/targets/x86_64-linux/include)
link_directories(/usr/local/cuda/targets/x86_64-linux/lib)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_61;code=sm_61;-gencode;arch=compute_60;code=sm_60;-gencode;arch=compute_62;code=sm_62)

file(GLOB_RECURSE PROJECT_SOURCES
        ${PROJECT_SOURCE_DIR}/include/*
        ${PROJECT_SOURCE_DIR}/src/*
        ${PARENT_DIR}/neural-network/include/common/*
        ${PARENT_DIR}/neural-network/src/common/*
        ${PARENT_DIR}/neural-network/include/logger/*
        ${PARENT_DIR}/neural-network/include/tensorrt-base/*
        ${PARENT_DIR}/neural-network/src/tensorrt-base/*
)

cuda_add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} )
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)

add_definitions(-O2 -pthread)