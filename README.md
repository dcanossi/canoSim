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

## Usage ##

canoSim++ has an integrated Cartesian mesh generator, which can be executed
by running the program with the `-mesh` flag. An input chart named
_meshInput.txt_ is mandatory.

After the mesh is created, all fields must be initialized using the _initFields.txt_
input file and running the executable with the `-init` flag. All boundary conditions
must be specified in that chart as well.

The last step is the solver run, which can be executed by passing the `-solve` flag
to the program. All the simulation parameters are inputted in the _control.txt_ file.

A typical simulation workflow consists of the following commands:

```
canoSim++ -mesh
canoSim++ -init
canoSim++ -solve
```

An example case is available in `examples/heat2D` illustrating the necessary input files.

<img width="836" height="371" alt="Image" src="https://github.com/user-attachments/assets/14b3d3c9-7a8f-4deb-9bfd-d7902999e850" />
