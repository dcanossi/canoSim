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

#include "mathFuncs.h"

__global__ void mathFuncs::matMul_d
(
    float* A,
    float* B,
    float* C,
    int m,
    int k,
    int n
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.z;

    if (row < m && col < n)
    {
        float sum = 0.0;
        for (int l = 0; l < k; l++)
        {
            sum += A[row * k + l] * B[l * n + col];
        }

        C[row * n + col] = sum;
    }
}

void mathFuncs::matMul_h(float* A, float* B, float* C, int m, int k, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float sum = 0.0;

            for (int l = 0; l < k; l++)
            {
                sum += A[i*k + l] * B[l*n + j];
            }

            C[i*n + j] = sum;
        }
    }
}