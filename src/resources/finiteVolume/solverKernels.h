/*─────────────────────────────────────────────────────────────────────────────┐
│       ╔═══╗   ╔═══╗   ╔═══╗   ╔═══╗   ╔═══╗   ╔═══╗   ╔═══╗   ╔══════╗       │
│       ║ c ║   ║ a ║   ║ n ║   ║ o ║   ║ S ║   ║ i ║   ║ m ║   ║ +  + ║       │
│       ╚═══╝   ╚═══╝   ╚═══╝   ╚═══╝   ╚═══╝   ╚═══╝   ╚═══╝   ╚══════╝       │
│                                                                              │
│                       ~ Computational Fluid Dynamics ~                       │
│                       High-Performance GPU Flow Solver                       │
│                                                                              │
│                                                                              │
│                                            Copyright (c) 2025 Dário Canossi  │
└─────────────────────────────────────────────────────────────────────────────*/

#ifndef solverKernels_H
#define solverKernels_H

#include "EulerSolver.h"

namespace solverKernels
{

// Heat capacity ratio (air)
constexpr float GAMMA = 1.4f;

// CUDA device functions for conversion to primitive variables
__host__ __device__ inline primitiveVars conservativeToPrimitive
(
    const conservativeVars& cons
);

// CUDA device functions for conversion to conservative variables
__host__ __device__ inline conservativeVars primitiveToConservative
(
    const primitiveVars& prim
);

// First-order upwind flux computation
__device__ inline faceFlux computeUpwindFlux
(
    const conservativeVars& consL,
    const conservativeVars& consR,
    float nx,
    float ny,
    float nz
);

// CUDA kernel for computing residuals
__global__ void computeResidual
(
    conservativeVars* U,
    conservativeVars* residual,
    int nx,
    int ny,
    int nz,
    float dx,
    float dy,
    float dz
);

// CUDA kernel for the application of wall boundary conditions
__global__ void applyBoundaryConditions
(
    conservativeVars* U,
    int nx,
    int ny,
    int nz
);

// CUDA kernel for first-order, forward Euler time integration
__global__ void integrateTime
(
    conservativeVars* U,
    conservativeVars* residual,
    float dt,
    int nx,
    int ny,
    int nz
);

}

#include "solverKernelsI.cu"

#endif