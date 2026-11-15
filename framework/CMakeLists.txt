cmake_minimum_required(VERSION 3.18)

add_library(telescope_framework STATIC
    src/parameter.cpp
    src/sensitivity.cpp
    src/control.cpp
    src/provenance.cpp
    src/fits_reader.cpp
    plugins/synthetic_linear.cpp
)
target_include_directories(telescope_framework PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/plugins)
target_link_libraries(telescope_framework PUBLIC telescope_numerics telescope_cpu)
target_compile_features(telescope_framework PUBLIC cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(telescope_framework PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(framework_demo apps/framework_demo.cpp)
target_link_libraries(framework_demo PRIVATE telescope_framework)

# Optional CUDA adapter: wraps the existing PhaseDiversitySolver.  It is a
# separate target so the CPU framework/demo still builds without CUDA.
if(TEL_BUILD_CUDA)
    add_library(telescope_framework_phase STATIC
        plugins/phase_diversity/phase_adapter.cpp
    )
    target_include_directories(telescope_framework_phase PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/plugins
    )
    target_link_libraries(telescope_framework_phase PUBLIC
        telescope_framework
        telescope_cuda
    )
endif()

if(TEL_BUILD_TESTS)
    add_executable(framework_tests tests/test_framework.cpp)
    target_link_libraries(framework_tests PRIVATE telescope_framework)
    add_test(NAME framework_tests COMMAND framework_tests)
endif()
