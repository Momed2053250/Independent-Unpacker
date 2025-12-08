cmake_minimum_required(VERSION 3.21)
project(Unpacker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# NVHPC C++ compiler (nvc++)
set(CMAKE_CXX_COMPILER /opt/nvhpc/2025/Linux_x86_64/25.1/compilers/bin/nvc++)
set(CMAKE_CUDA_COMPILER /opt/nvhpc/2025/Linux_x86_64/25.1/compilers/bin/nvcc)
# Enable CUDA support for nvc++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -cuda")

enable_language(CUDA)

# Alpaka backends
set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE  ON  CACHE BOOL "CPU serial" FORCE)  # CHANGED: OFF -> ON
set(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OFF  CACHE BOOL "CPU OpenMP" FORCE)
set(alpaka_ACC_GPU_CUDA_ENABLE          OFF  CACHE BOOL "GPU CUDA" FORCE)
set(alpaka_ACC_GPU_HIP_ENABLE           OFF CACHE BOOL "GPU HIP" FORCE)
set(alpaka_ACC_GPU_SYCL_ENABLE          OFF CACHE BOOL "GPU SYCL" FORCE)

# Alpaka
add_subdirectory("$ENV{HOME}/alpaka" "${CMAKE_BINARY_DIR}/alpaka_build")

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(USE_OPENMP TRUE)
endif()

# Executable
add_executable(unpacker src/Unpacker.cpp)
target_include_directories(unpacker PRIVATE ${CMAKE_SOURCE_DIR}/include)
if(alpaka_ACC_GPU_CUDA_ENABLE)  # CHANGED: ALPAKA_ACC_GPU_CUDA_ENABLE -> alpaka_ACC_GPU_CUDA_ENABLE
    target_compile_definitions(unpacker PRIVATE ALPAKA_CUDA_ENABLED=1)  # CHANGED: USE_CUDA=1 -> ALPAKA_CUDA_ENABLED=1
endif()
if(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE)  # ADDED: CPU backend definition
    target_compile_definitions(unpacker PRIVATE ALPAKA_CPU_SERIAL_ENABLED=1)
endif()
target_link_libraries(unpacker PRIVATE alpaka::alpaka)
if(USE_OPENMP)
    target_link_libraries(unpacker PRIVATE OpenMP::OpenMP_CXX)
endif()

# CUDA options
set_target_properties(unpacker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_property(TARGET unpacker PROPERTY LANGUAGES CXX CUDA)

# Warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(unpacker PRIVATE -Wall -Wextra -Wpedantic)
elseif (MSVC)
    target_compile_options(unpacker PRIVATE /W4)
endif()

# Integration test using real PackedData.root (no Catch2 needed)
# COMMENTED OUT to avoid ROOT warnings
# find_package(ROOT REQUIRED COMPONENTS Core Tree)

# add_executable(test_unpacker_data
#     test/Unpacker_data_integration.cpp
# )

# target_link_libraries(test_unpacker_data
#     PRIVATE
#         alpaka::alpaka            # CHANGED: unpacker -> alpaka::alpaka
#         ROOT::Core
#         ROOT::Tree
# )
# target_include_directories(test_unpacker_data PRIVATE ${CMAKE_SOURCE_DIR}/include)  # ADDED