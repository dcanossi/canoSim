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
    std::cout << "\nKernel sum = " << result << std::endl;

    return 0;
}