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

#ifndef mathFuncs_H
#define mathFuncs_H

#include <array>
#include <vector>
#include <cuda/std/span>

namespace mathFuncs
{

// Kernel for vector reduction
template<int blockSize>
__global__ void reduce
(
    cuda::std::span<const int> data,
    cuda::std::span<int> result
);

// Naive matrix multiplication on device
__global__ void matMul_d(float* A, float* B, float* C, int m, int k, int n);

// Main function for vector reduction
template<typename Type>
Type vecReduce(const std::vector<Type>& vec);

// Fill a N x N matrix
template<typename Type, int nRows, int nColumns>
void fillMatrix(float* mat);

// Matrix multiplication on host
void matMul_h(float* A, float* B, float* C, int m, int k, int n);

}

#include "mathFuncsT.cu"

#endif