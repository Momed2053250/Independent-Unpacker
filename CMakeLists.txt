cmake_minimum_required(VERSION 3.21)
project(Unpacker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ========================
# NVHPC compilers (full paths)
# ========================
set(CMAKE_CXX_COMPILER /opt/nvhpc/2025/Linux_x86_64/25.1/compilers/bin/nvc++)
set(CMAKE_CUDA_COMPILER /opt/nvhpc/2025/Linux_x86_64/25.1/compilers/bin/nvcc)
enable_language(CUDA)

# ========================
# Alpaka backends
# ========================
set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE  ON  CACHE BOOL "CPU serial" FORCE)
set(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE ON  CACHE BOOL "CPU OpenMP" FORCE)
set(alpaka_ACC_GPU_CUDA_ENABLE          ON  CACHE BOOL "GPU CUDA" FORCE)
set(alpaka_ACC_GPU_HIP_ENABLE           OFF CACHE BOOL "GPU HIP" FORCE)
set(alpaka_ACC_GPU_SYCL_ENABLE          OFF CACHE BOOL "GPU SYCL" FORCE)

# ========================
# Alpaka library
# ========================
add_subdirectory("$ENV{HOME}/alpaka" "${CMAKE_BINARY_DIR}/alpaka_build")

# ========================
# OpenMP
# ========================
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(USE_OPENMP TRUE)
endif()

# ========================
# Executable
# ========================
add_executable(unpacker
    src/Unpacker.cpp
)

target_include_directories(unpacker PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Compile definitions for CUDA backend
if(ALPAKA_ACC_GPU_CUDA_ENABLE)
    target_compile_definitions(unpacker PRIVATE USE_CUDA=1)
endif()

target_link_libraries(unpacker PRIVATE alpaka::alpaka)
if(USE_OPENMP)
    target_link_libraries(unpacker PRIVATE OpenMP::OpenMP_CXX)
endif()

# ========================
# CUDA options
# ========================
set_target_properties(unpacker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_property(TARGET unpacker PROPERTY LANGUAGES CXX CUDA)

# ========================
# Warnings
# ========================
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(unpacker PRIVATE -Wall -Wextra -Wpedantic)
elseif (MSVC)
    target_compile_options(unpacker PRIVATE /W4)
endif()
