cmake_minimum_required(VERSION 3.18)
project(telescope_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(TEL_BUILD_CUDA "Build CUDA targets (forward model, solver, GPU tests)" ON)
option(TEL_BUILD_TESTS "Build unit tests" ON)

# RTX 3050 = sm_86。用户显式 -DCMAKE_CUDA_ARCHITECTURES=... 时不再覆盖；
# 不再把 CMake 自动推断的 52 强行改掉，避免在多卡/新卡机器上编译出
# 不兼容的 cubin。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures (e.g. 86, 89)")
endif()

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
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(telescope_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

# GCC 8 的 std::filesystem 在单独的库里；GCC 9+ 通常不需要。
find_library(STDCXXFS_LIBRARY stdc++fs)
if(STDCXXFS_LIBRARY)
    target_link_libraries(telescope_cpu PUBLIC ${STDCXXFS_LIBRARY})
endif()

# ------------------------------------------------------------
# 2. CUDA 可选层：找不到 CUDA 或 NLopt 时仍可构建 CPU 测试。
# ------------------------------------------------------------
if(TEL_BUILD_CUDA)
    include(CheckLanguage)
    # 有些机器装了 CUDA 但 nvcc 不在 PATH（只在 /usr/local/cuda/bin）。
    # 先显式找一次，避免把“已安装但没进 PATH”误判成“没有 CUDA”。
    if(NOT CMAKE_CUDA_COMPILER)
        find_program(TEL_CUDA_COMPILER nvcc
            PATHS /usr/local/cuda/bin /usr/local/cuda-12.4/bin /usr/local/cuda-12/bin
            NO_DEFAULT_PATH)
        if(TEL_CUDA_COMPILER)
            set(CMAKE_CUDA_COMPILER "${TEL_CUDA_COMPILER}" CACHE FILEPATH "CUDA compiler")
        endif()
    endif()
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(WARNING "CUDA compiler not found; building CPU-only targets "
                        "(set CMAKE_CUDA_COMPILER or add nvcc to PATH)")
        set(TEL_BUILD_CUDA OFF)
    else()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        find_package(Threads REQUIRED)
        find_library(NLOPT_LIBRARY nlopt)
        if(NOT NLOPT_LIBRARY)
            message(WARNING "NLopt not found; building CPU-only targets "
                            "(install libnlopt-cxx-dev to enable inverse_solve)")
            set(TEL_BUILD_CUDA OFF)
        endif()
    endif()
endif()

if(TEL_BUILD_CUDA)
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
    if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
        target_compile_options(telescope_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
        )
    endif()

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
endif()

# ------------------------------------------------------------
# 3. 测试
# ------------------------------------------------------------
if(TEL_BUILD_TESTS)
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

    if(TEL_BUILD_CUDA)
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
    endif()
endif()

if(TEL_BUILD_CUDA)
    message(STATUS "telescope_gpu: CUDA arch = ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "telescope_gpu: CPU-only build")
endif()
