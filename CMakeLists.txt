cmake_minimum_required(VERSION 3.31)
cmake_policy(SET CMP0167 OLD)

project(Unpacker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- NVHPC (optional) ---
set(CMAKE_CXX_COMPILER /opt/nvhpc/2025/Linux_x86_64/25.1/compilers/bin/nvc++)
set(CMAKE_CUDA_COMPILER /opt/nvhpc/2025/Linux_x86_64/25.1/compilers/bin/nvcc)
enable_language(CUDA)

# Alpaka backends
set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE  OFF)
set(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OFF)
set(alpaka_ACC_GPU_CUDA_ENABLE          ON)
set(alpaka_ACC_GPU_HIP_ENABLE           OFF)
set(alpaka_ACC_GPU_SYCL_ENABLE          OFF)

add_subdirectory("$ENV{HOME}/alpaka" "${CMAKE_BINARY_DIR}/alpaka_build")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(USE_OPENMP TRUE)
endif()

# Library (no main)
alpaka_add_library(unpacker_lib src/Unpacker.cpp)
target_include_directories(unpacker_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(unpacker_lib PUBLIC NO_MAIN_IN_UNPACKER)
target_link_libraries(unpacker_lib PUBLIC alpaka::alpaka)
if(USE_OPENMP)
  target_link_libraries(unpacker_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Executable (optional)
alpaka_add_executable(unpacker src/Unpacker.cpp)
target_include_directories(unpacker PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(unpacker PUBLIC alpaka::alpaka)
if(USE_OPENMP)
  target_link_libraries(unpacker PUBLIC OpenMP::OpenMP_CXX)
endif()

# Test
alpaka_add_executable(test_unpack test/test_unpack.cpp)
target_link_libraries(test_unpack PRIVATE unpacker_lib alpaka::alpaka)
if(USE_OPENMP)
  target_link_libraries(test_unpack PRIVATE OpenMP::OpenMP_CXX)
endif()
