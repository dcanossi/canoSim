# canoSim++

**High-Performance GPU-based CFD Solver.**

Solves partial differential equations for Computational Fluid Dynamics
in a finite-volume mesh.

Written in C++ and accelerated using CUDA kernels. Includes a Cartesian mesh
generator.

Adopts modern industry practices:

- CMake build
- C++17 standard
- Extensive use of the STL library
- NVIDIA nvcc compiler v12.9

## Compilation ##

First, you need to install all necessary dependencies: the CMake build generator
(minimum version: v4.0), a C++ compiler with support for the C++17 standard and
CUDA toolkit 12.

Then, follow these steps to compile canoSim++:

1. In the root, set a separate build directory: `mkdir build && cd build`
2. Generate a Makefile from the main CMakeLists.txt: `cmake ..`
3. Build the software: `cmake --build .`

## Usage ##

canoSim++ has an integrated Cartesian mesh generator, which can be executed
by running the program with the `-mesh` flag. An input chart named
_meshInput.txt_ is mandatory.

After the mesh is created, all fields must be initialized via the _initFields.txt_
input file and running the executable with the `-init` flag. All boundary conditions
must be specified in that chart as well.

The last step is the solver run, which can be executed by passing the `-solve` flag
to the program. All the run-time simulation parameters are inputted in the _controls.txt_ file.

A typical simulation workflow consists of the following commands:

```
canoSim++ -mesh
canoSim++ -init
canoSim++ -solve
```

After running a simulation, a mesh VTK file will be written to a 'mesh' folder, while the resulting
fields will be outputted (also in VTK format) to a 'solution' folder. An example case is available
in `examples/euler3D` illustrating the necessary input files.

<img width="836" height="371" alt="Image" src="https://github.com/user-attachments/assets/14b3d3c9-7a8f-4deb-9bfd-d7902999e850" />
