#──────────────────────────────────────────────────────────────────────────────#
#       ╔═══╗   ╔═══╗   ╔═══╗   ╔═══╗   ╔═══╗   ╔═══╗   ╔═══╗   ╔══════╗       #
#       ║ c ║   ║ a ║   ║ n ║   ║ o ║   ║ S ║   ║ i ║   ║ m ║   ║ +  + ║       #
#       ╚═══╝   ╚═══╝   ╚═══╝   ╚═══╝   ╚═══╝   ╚═══╝   ╚═══╝   ╚══════╝       #
#                                                                              #
#                       ~ Computational Fluid Dynamics ~                       #
#                       High-Performance GPU Flow Solver                       #
#                                                                              #
#                                                                              #
#                                            Copyright (c) 2025 Dário Canossi  #
#──────────────────────────────────────────────────────────────────────────────#

# Project CMake configuration with C++ and CUDA support
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)
project(canoSim VERSION 0.1 LANGUAGES CXX CUDA)

message("\n-- CMake build configuration for canoSim++ v${CMAKE_PROJECT_VERSION}")

# Set C++-17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Additional compilation flags
add_compile_options(-Wall -Wextra -Wno-deprecated-gpu-targets)

# Set library and binary directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# List of targets to build
add_library(basic STATIC "src/basic/clock.cpp")

add_library(resources STATIC
    "src/resources/math/mathFuncs.cu"
    "src/resources/mesh/mesh.cpp"
)

add_executable(canoSim "solvers/canoSim.cpp")
target_include_directories(canoSim PUBLIC
    "src/basic"
    "src/resources/mesh"
)

add_executable(matrixMultiply "solvers/matrixMultiply.cu")
target_include_directories(matrixMultiply PUBLIC
    "src/resources/math"
)

target_link_libraries(canoSim basic resources)
target_link_libraries(matrixMultiply resources)

# CMake build properties
set_source_files_properties("solvers/canoSim.cpp" PROPERTIES LANGUAGE CUDA)

set_target_properties(basic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(resources PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(resources PROPERTIES CUDA_ARCHITECTURES 60)
set_target_properties(matrixMultiply PROPERTIES CUDA_ARCHITECTURES 60)