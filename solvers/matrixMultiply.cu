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

#include <iostream>
#include <numeric>
#include <vector>
#include "mathFuncs.h"

// Wrap the cudaMalloc call to avoid the need for pointer-to-pointers
template<typename T>
cudaError_t cudaAlloc(T*& d_p, size_t elements)
{
    return cudaMalloc((void**)&d_p, elements*sizeof(T));
}

int main()
{
    // VECTOR REDUCTION
    const int N = 1000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 1);

    const int result = mathFuncs::vecReduce<int>(data);
    // std::cout << "\nKernel sum = " << result << "\n" << std::endl;

    // MATRIX MULTIPLICATION
    constexpr const int m = 64;
    constexpr const int k = 128;
    constexpr const int n = 32;

    constexpr const int size_A = m*k;
    constexpr const int size_B = k*n;
    constexpr const int size_C = m*n;

    float matA_h[size_A];
    float matB_h[size_B];
    float matC_h[size_C];

    srand(time(nullptr));
    mathFuncs::fillMatrix<float, m, k>(matA_h);
    mathFuncs::fillMatrix<float, k, n>(matB_h);

    float *matA_d, *matB_d, *matC_d;
    cudaAlloc(matA_d, size_A);
    cudaAlloc(matB_d, size_B);
    cudaAlloc(matC_d, size_C);

    cudaMemcpy(matA_d, matA_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matB_d, matB_h, size_B, cudaMemcpyHostToDevice);

    // Grid and block dimensions
    constexpr const int blockSize = 8;
    dim3 blockDim = {blockSize, blockSize};
    dim3 gridDim = (std::ceil(n/blockSize), std::ceil(m/blockSize));

    auto getTime = []()
    {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);

        return ts.tv_sec + 1e-9*ts.tv_nsec;
    };

    std::cout << "\nPerforming matrix multiplication on the CPU...";
    double startTime = getTime();
    mathFuncs::matMul_h(matA_h, matB_h, matC_h, m, k, n);
    double endTime = getTime();
    double cpuTime = endTime - startTime;
    std::cout << "\nDone in " << cpuTime*1e6 << " microseconds\n";

    std::cout << "\nPerforming matrix multiplication on the GPU...";
    startTime = getTime();
    mathFuncs::matMul_d<<<gridDim, blockDim>>>(matA_d, matB_d, matC_d, m, k, n);
    endTime = getTime();
    cpuTime = endTime - startTime;
    std::cout << "\nDone in " << cpuTime*1e6 << " microseconds" << std::endl;

    return 0;
}