cmake_minimum_required(VERSION 3.24)
project(sixi_asi VERSION 1.0.0 LANGUAGES CXX)
# Development builds carry a suffix; a release sets it to "" (one place for the version: here)
set(SIXI_VERSION_SUFFIX "-dev" CACHE STRING "Appended to the version the binaries report")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SIXI_ENABLE_CUDA "Build native sixi-asi CUDA kernels" ON)
option(SIXI_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(SIXI_BUILD_TESTS "Build sixi-asi unit, differential, and integration test suite" ON)
option(SIXI_BUILD_EXAMPLES "Build the examples (library use)" ON)

# Determinism-safe compiler flags (Section 4: no fast-math flags that break IEEE-754 determinism)
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
)

if(SIXI_ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

set(SIXI_ASI_CORE_SOURCES
    src/common/common.cpp
    src/io/gguf.cpp
    src/io/manifest.cpp
    src/io/sidecar.cpp
    src/io/io_reader.cpp
    src/memory/memory_planner.cpp
    src/memory/offload.cpp
    src/core/numeric.cpp
    src/core/core_execution.cpp
    src/core/constraint.cpp
    src/speculative/speculative.cpp
    src/server/server.cpp
)

if(SIXI_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 110)
        endif()
        list(APPEND SIXI_ASI_CORE_SOURCES src/cuda/sixi_kernels.cu)
        add_compile_definitions(SIXI_HAS_CUDA=1)
    endif()
endif()

add_library(sixi_asi STATIC ${SIXI_ASI_CORE_SOURCES})

target_include_directories(sixi_asi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
        $<INSTALL_INTERFACE:include>
)
target_compile_definitions(sixi_asi PUBLIC SIXI_ASI_VERSION="${PROJECT_VERSION}${SIXI_VERSION_SUFFIX}")
target_link_libraries(sixi_asi
    PUBLIC
        Threads::Threads
        OpenSSL::Crypto
        OpenSSL::SSL
        ${CMAKE_DL_LIBS}
)

if(CMAKE_CUDA_COMPILER AND SIXI_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(sixi_asi PUBLIC CUDA::cudart)
endif()

# Primary sixi-asi CLI, Server & Benchmark Executables
add_executable(sixi-cli apps/sixi_cli.cpp)
target_link_libraries(sixi-cli PRIVATE sixi_asi)

add_executable(sixi-server apps/sixi_server_main.cpp)
target_link_libraries(sixi-server PRIVATE sixi_asi)

add_executable(sixi-bench apps/sixi_bench.cpp)
target_link_libraries(sixi-bench PRIVATE sixi_asi)

if(SIXI_BUILD_EXAMPLES)
    add_executable(example_generate examples/generate.cpp)
    target_link_libraries(example_generate PRIVATE sixi_asi)
endif()

# Install: the three programs, the static library with its headers, and a CMake package
# (find_package(sixi_asi) -> sixi_asi::sixi_asi). The llama.cpp backend is loaded at run time and is not part of it.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS sixi_asi sixi-server sixi-cli sixi-bench EXPORT sixi_asiTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/sixi_asi DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT sixi_asiTargets NAMESPACE sixi_asi:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sixi_asi)
if(CMAKE_CUDA_COMPILER AND SIXI_ENABLE_CUDA)
    set(SIXI_ASI_WITH_CUDA ON)
else()
    set(SIXI_ASI_WITH_CUDA OFF)
endif()
configure_package_config_file(cmake/sixi_asiConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/sixi_asiConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sixi_asi)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/sixi_asiConfigVersion.cmake COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sixi_asiConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/sixi_asiConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sixi_asi)

# Automated Test Suite (Section 11)
if(SIXI_BUILD_TESTS)
    enable_testing()

    set(SIXI_TEST_TARGETS
        test_io_and_overflow
        test_routing_and_numerics
        test_determinism_and_speculative
        test_memory_and_offload
        test_server_and_tools
        test_constraint
    )

    if(CMAKE_CUDA_COMPILER AND SIXI_ENABLE_CUDA)
        list(APPEND SIXI_TEST_TARGETS test_cuda_kernels)
    endif()

    foreach(test_name IN LISTS SIXI_TEST_TARGETS)
        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE sixi_asi)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()
