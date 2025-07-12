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
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <thrust/device_vector.h>

template<int blockSize>
__global__ void mathFuncs::reduce
(
    cuda::std::span<const int> data,
    cuda::std::span<int> result
)
{
    // CUB library struct to perform thread and block reduction
    using BlockReduce = cub::BlockReduce<int, blockSize>;

    __shared__ typename BlockReduce::TempStorage tStorage;

    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    int sum = 0;
    if (index < data.size())
    {
        sum += data[index];
    }

    // Thread reduction for each block
    sum = BlockReduce(tStorage).Sum(sum);

    // Inter-block reduction
    if (threadIdx.x == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> atResult
        (
            result.front()
        );
        atResult.fetch_add(sum, cuda::memory_order_relaxed);
    }
}

template<typename Type>
Type mathFuncs::vecReduce(const std::vector<Type>& vec)
{
    // Allocate and initialize input data
    const int N = vec.size();
    thrust::device_vector<int> data = vec;

    // Allocate output data
    thrust::device_vector<int> d_result(1);

    // Compute the sum reduction of `data` using a kernel
    constexpr int blockSize = 256;
    const int nBlocks = cuda::ceil_div(N, blockSize);
    reduce<blockSize><<<nBlocks, blockSize>>>
    (
        cuda::std::span<const int>
        (
            thrust::raw_pointer_cast(data.data()), data.size()
        ),
        cuda::std::span<int>(thrust::raw_pointer_cast(d_result.data()), 1)
    );

    const auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        return {};
    }

    return d_result[0];
}

template<typename Type, int nRows, int nColumns>
void mathFuncs::fillMatrix(float* mat)
{
    std::cout << "Generating matrix [" << nRows << "x" << nColumns << "]";

    for (int i = 0; i < nRows * nColumns; i++)
    {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
        std::cout << " " << mat[i];
    }
    std::cout << std::endl;
}