cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_FLAGS "-arch=sm_60")
project(MatrixMul CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(SOCURCES "")

add_executable(MatrixMul src/main.cu src/main.cuh src/common.cuh src/cudaMultiplier_own.cu src/cudaMultiplier_own.cuh src/cudaMultiplier_standard.cuh src/cudaMultiplier_standard.cu)

set_target_properties(
        MatrixMul
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)