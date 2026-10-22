cmake_minimum_required(VERSION 3.18)
project(telescope_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

# 脚本 1: forward_generate
add_executable(forward_generate
    src/forward_generate.cu
    src/forward_model.cu
)
target_link_libraries(forward_generate PRIVATE
    CUDA::cudart CUDA::cufft
)
set_target_properties(forward_generate PROPERTIES CUDA_ARCHITECTURES "86")

# 脚本 2: inverse_solve
add_executable(inverse_solve
    src/inverse_solve.cu
    src/forward_model.cu
)
target_link_libraries(inverse_solve PRIVATE
    CUDA::cudart CUDA::cufft nlopt
)
set_target_properties(inverse_solve PROPERTIES CUDA_ARCHITECTURES "86")