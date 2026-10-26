cmake_minimum_required(VERSION 3.18)
project(telescope_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 3050 = sm_86。换机器时用 -DCMAKE_CUDA_ARCHITECTURES=...
# CMake 可能已经缓存了一个默认值（如 52），所以这里也覆盖这种默认情况。
if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_library(NLOPT_LIBRARY nlopt REQUIRED)

# ------------------------------------------------------------
# 1. CPU 核心：不依赖 CUDA，可在无 GPU 环境编译和测试。
# ------------------------------------------------------------
add_library(telescope_cpu STATIC
    src/zernike.cpp
    src/fft_cpu.cpp
    src/fits_io.cpp
    src/cpu_reference.cpp
    src/config.cpp
)
target_include_directories(telescope_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(telescope_cpu PUBLIC cxx_std_17)
target_compile_options(telescope_cpu PRIVATE -Wall -Wextra -Wpedantic)

# GCC 8 的 std::filesystem 在单独的库里；GCC 9+ 通常不需要。
find_library(STDCXXFS_LIBRARY stdc++fs)
if(STDCXXFS_LIBRARY)
    target_link_libraries(telescope_cpu PUBLIC ${STDCXXFS_LIBRARY})
endif()

# ------------------------------------------------------------
# 2. CUDA 核心：正向模型 + 相位差反演。
# ------------------------------------------------------------
add_library(telescope_cuda STATIC
    src/forward_model.cu
    src/phase_diversity.cu
)
target_link_libraries(telescope_cuda PUBLIC
    telescope_cpu
    CUDA::cudart
    CUDA::cufft
    ${NLOPT_LIBRARY}
    Threads::Threads
)
set_target_properties(telescope_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
target_compile_options(telescope_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

# ------------------------------------------------------------
# 3. 可执行程序
# ------------------------------------------------------------
add_executable(forward_generate apps/forward_generate.cu)
target_link_libraries(forward_generate PRIVATE telescope_cuda)
set_target_properties(forward_generate PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

add_executable(inverse_solve apps/inverse_solve.cu)
target_link_libraries(inverse_solve PRIVATE telescope_cuda)
set_target_properties(inverse_solve PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# ------------------------------------------------------------
# 4. 测试
# ------------------------------------------------------------
enable_testing()

add_executable(telescope_tests
    tests/test_main.cpp
    tests/test_zernike.cpp
    tests/test_fft.cpp
    tests/test_fits.cpp
    tests/test_config.cpp
)
target_link_libraries(telescope_tests PRIVATE telescope_cpu)
target_include_directories(telescope_tests PRIVATE tests)
add_test(NAME cpu_unit COMMAND telescope_tests)

add_executable(telescope_gpu_tests
    tests/test_main.cpp
    tests/test_cuda.cu
)
target_link_libraries(telescope_gpu_tests PRIVATE telescope_cuda)
target_include_directories(telescope_gpu_tests PRIVATE tests)
set_target_properties(telescope_gpu_tests PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
add_test(NAME gpu_unit COMMAND telescope_gpu_tests)

# 让在 build/ 下运行时也能找到 configs/default.conf。
# 用 custom target 而不是 configure 时的 file(COPY)，这样修改源码里的
# configs/*.conf 后，下一次 build 会自动同步，不会用到 build/ 里的旧副本。
add_custom_target(copy_configs ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/configs
            ${CMAKE_BINARY_DIR}/configs
    COMMENT "Syncing configs/ to build directory")
add_dependencies(forward_generate copy_configs)
add_dependencies(inverse_solve copy_configs)

message(STATUS "telescope_gpu: CUDA arch = ${CMAKE_CUDA_ARCHITECTURES}")
