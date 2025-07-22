# canoSim++

**High-Performance GPU-based CFD Solver.**

Solves partial differential equations for Computational Fluid Dynamics
in a finite-volume mesh.

Written in C++ and accelerated using CUDA kernels. Includes a Cartesian mesh
generator.

Adopts modern industry practices:

- CMake build
- C++-17 standard
- Extensive use of the STL library
- NVIDIA nvcc compiler v12.9