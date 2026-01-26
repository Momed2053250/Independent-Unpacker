cmake_minimum_required(VERSION 3.31)

# Fix Boost / CMP0167 warning
cmake_policy(SET CMP0167 OLD)

project(Unpacker LANGUAGES CXX CUDA)

# =========================
# Global C++ / CUDA setup
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# NVHPC compilers
set(CMAKE_CXX_COMPILER /opt/nvhpc/2025/Linux_x86_64/25.1/compilers/bin/nvc++)
set(CMAKE_CUDA_COMPILER /opt/nvhpc/2025/Linux_x86_64/25.1/compilers/bin/nvcc)

# Enable CUDA in nvc++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -cuda")

enable_language(CUDA)

# =========================
# Alpaka configuration
# =========================
set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE  OFF)
set(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OFF)
set(alpaka_ACC_GPU_CUDA_ENABLE          ON)
set(alpaka_ACC_GPU_HIP_ENABLE           OFF)
set(alpaka_ACC_GPU_SYCL_ENABLE          OFF)

add_subdirectory("$ENV{HOME}/alpaka" "${CMAKE_BINARY_DIR}/alpaka_build")

# =========================
# OpenMP
# =========================
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(USE_OPENMP TRUE)
endif()

# =========================
# Library (NO main)
# =========================
add_library(unpacker_lib src/Unpacker.cpp)
target_include_directories(unpacker_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(unpacker_lib PUBLIC NO_MAIN_IN_UNPACKER)

if(alpaka_ACC_GPU_CUDA_ENABLE)
    target_compile_definitions(unpacker_lib PUBLIC ALPAKA_CUDA_ENABLED=1)
endif()

if(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE)
    target_compile_definitions(unpacker_lib PUBLIC ALPAKA_CPU_SERIAL_ENABLED=1)
endif()

target_link_libraries(unpacker_lib PUBLIC alpaka::alpaka)

if(USE_OPENMP)
    target_link_libraries(unpacker_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(unpacker_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_property(TARGET unpacker_lib PROPERTY LANGUAGES CXX CUDA)

# =========================
# Main executable
# =========================
add_executable(unpacker src/Unpacker.cpp)
target_include_directories(unpacker PUBLIC ${CMAKE_SOURCE_DIR}/include)

if(alpaka_ACC_GPU_CUDA_ENABLE)
    target_compile_definitions(unpacker PUBLIC ALPAKA_CUDA_ENABLED=1)
endif()

if(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE)
    target_compile_definitions(unpacker PUBLIC ALPAKA_CPU_SERIAL_ENABLED=1)
endif()

target_link_libraries(unpacker PUBLIC alpaka::alpaka)

if(USE_OPENMP)
    target_link_libraries(unpacker PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(unpacker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_property(TARGET unpacker PROPERTY LANGUAGES CXX CUDA)

# =========================
# Test executable (NO ROOT)
# =========================
add_executable(test_unpack test/test_unpack.cpp)
target_link_libraries(test_unpack PRIVATE unpacker_lib alpaka::alpaka)

if(USE_OPENMP)
    target_link_libraries(test_unpack PRIVATE OpenMP::OpenMP_CXX)
endif()

# =========================
# Warnings
# =========================
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(unpacker_lib PRIVATE -Wall -Wextra -Wpedantic)
elseif (MSVC)
    target_compile_options(unpacker_lib PRIVATE /W4)
endif()
