cmake_minimum_required(VERSION 3.18)
project(telescope_numerics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TEL_NUMERICS_BUILD_TESTS "Build GTest unit tests" ON)
option(TEL_NUMERICS_BUILD_BENCH "Build Google Benchmark targets" ON)
option(TEL_NUMERICS_BUILD_PYTHON "Build pybind11 module" ON)
option(TEL_NUMERICS_ENABLE_ASAN "Enable AddressSanitizer" OFF)

include(FetchContent)
enable_testing()

if(TEL_NUMERICS_ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# ---- Eigen (header-only) --------------------------------------------------
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(Eigen3_FOUND)
    set(TEL_EIGEN_TARGET Eigen3::Eigen)
else()
    FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(eigen)
    set(TEL_EIGEN_TARGET eigen)
endif()

add_library(telescope_numerics src/numerics.cpp)
target_include_directories(telescope_numerics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(telescope_numerics PUBLIC ${TEL_EIGEN_TARGET})
target_compile_features(telescope_numerics PUBLIC cxx_std_17)
set_target_properties(telescope_numerics PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- GTest ----------------------------------------------------------------
if(TEL_NUMERICS_BUILD_TESTS)
    # On WSL, find_package(GTest) can pick up a Windows conda installation
    # (gtest.lib), which is not link-compatible with Linux.  Default to a
    # pinned source build; set TEL_NUMERICS_USE_SYSTEM_GTEST=ON to override.
    option(TEL_NUMERICS_USE_SYSTEM_GTEST "Use a system GTest instead of FetchContent" OFF)
    if(TEL_NUMERICS_USE_SYSTEM_GTEST)
        # 不走 find_package(GTest)：CMake 的 FindGTest 内部还会执行一次
        # config 模式查找，而 WSL 的 PATH 里有 Windows conda，会把
        # /mnt/c/.../GTestConfig.cmake（MSVC ABI）搜出来。这里只查 Linux
        # 系统目录，保证拿到的是 /usr/lib 下的 .a。
        find_path(TEL_GTEST_INCLUDE_DIR gtest/gtest.h
            PATHS /usr/include /usr/local/include
            NO_DEFAULT_PATH)
        find_library(TEL_GTEST_LIBRARY gtest
            PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
            NO_DEFAULT_PATH)
        find_library(TEL_GTEST_MAIN_LIBRARY gtest_main
            PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
            NO_DEFAULT_PATH)
        if(TEL_GTEST_INCLUDE_DIR AND TEL_GTEST_LIBRARY AND TEL_GTEST_MAIN_LIBRARY)
            find_package(Threads REQUIRED)
            add_library(tel_gtest_main INTERFACE)
            target_include_directories(tel_gtest_main INTERFACE
                ${TEL_GTEST_INCLUDE_DIR})
            target_link_libraries(tel_gtest_main INTERFACE
                ${TEL_GTEST_MAIN_LIBRARY} ${TEL_GTEST_LIBRARY}
                Threads::Threads)
            set(TEL_GTEST_TARGET tel_gtest_main)
        else()
            message(WARNING "System GTest not found under /usr; "
                            "falling back to FetchContent")
            FetchContent_Declare(googletest
                URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
            set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
            FetchContent_MakeAvailable(googletest)
            set(TEL_GTEST_TARGET gtest_main)
        endif()
    else()
        FetchContent_Declare(googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        set(TEL_GTEST_TARGET gtest_main)
    endif()
    add_executable(numerics_tests tests/test_numerics.cpp)
    target_link_libraries(numerics_tests PRIVATE telescope_numerics ${TEL_GTEST_TARGET})
    if(TEL_NUMERICS_ENABLE_ASAN)
        # ASan + gtest_discover_tests can abort during the discovery run under
        # WSL; register one CTest target and let it run all tests instead.
        add_test(NAME numerics_tests COMMAND numerics_tests)
    else()
        include(GoogleTest)
        gtest_discover_tests(numerics_tests)
    endif()
endif()

# ---- Google Benchmark -----------------------------------------------------
if(TEL_NUMERICS_BUILD_BENCH)
    find_package(benchmark QUIET)
    if(NOT benchmark_FOUND)
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(googlebenchmark
            URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(googlebenchmark)
    endif()
    add_executable(numerics_bench bench/bench_numerics.cpp)
    target_link_libraries(numerics_bench PRIVATE telescope_numerics benchmark::benchmark)
endif()

# ---- pybind11 -------------------------------------------------------------
if(TEL_NUMERICS_BUILD_PYTHON)
    find_program(TEL_PYTHON_EXECUTABLE NAMES python3 python)
    if(TEL_PYTHON_EXECUTABLE)
        # pybind11 3.x 默认走 FindPython；显式指定解释器，避免在 WSL 上
        # 把 /usr/bin/python3（3.8）和 venv 里的 3.12 混用。
        set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
        set(Python_EXECUTABLE "${TEL_PYTHON_EXECUTABLE}" CACHE FILEPATH "" FORCE)
        execute_process(
            COMMAND ${TEL_PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE TEL_PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(TEL_PYBIND11_CMAKE_DIR)
            list(APPEND CMAKE_PREFIX_PATH "${TEL_PYBIND11_CMAKE_DIR}")
        endif()
    endif()
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        FetchContent_Declare(pybind11
            URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(pybind11)
    endif()
    pybind11_add_module(telescope_numerics_py bindings/numerics_py.cpp)
    target_link_libraries(telescope_numerics_py PRIVATE telescope_numerics)
    set_target_properties(telescope_numerics_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
endif()
