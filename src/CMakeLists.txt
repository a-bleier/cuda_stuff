set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "-arch=sm_60")
set(SOCURCES "")


include_directories(${cudaTest_SOURCE_DIR}/inc)
add_executable(cudaTest main.cu cudaMultiplier_own.cu cudaMultiplier_standard.cu)

set_target_properties(
        cudaTest
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

