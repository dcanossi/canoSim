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

#include "solverKernels.h"

__global__ void solverKernels::computeResidual
(
    conservativeVars* U,
    conservativeVars* residual,
    int nx,
    int ny,
    int nz,
    float dx,
    float dy,
    float dz
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Check bounds (excluding ghost cells)
    if
    (
        i < 1 || i >= nx - 1
     || j < 1 || j >= ny - 1
     || k < 1 || k >= nz - 1
    ) return;

    int idx = i + j * nx + k * nx * ny;

    residual[idx] = conservativeVars(0, 0, 0, 0, 0);

    float volume = dx * dy * dz;

    // x-direction faces (West and East)
    int idxW = (i - 1) + j * nx + k * nx * ny;
    int idxE = (i + 1) + j * nx + k * nx * ny;

    faceFlux fluxW = computeUpwindFlux(U[idxW], U[idx], 1.0f, 0.0f, 0.0f);
    float areaX = dy * dz;

    residual[idx].rho -= fluxW.mass * areaX / volume;
    residual[idx].rhoU -= fluxW.momX * areaX / volume;
    residual[idx].rhoV -= fluxW.momY * areaX / volume;
    residual[idx].rhoW -= fluxW.momZ * areaX / volume;
    residual[idx].rhoE -= fluxW.energy * areaX / volume;

    faceFlux fluxE = computeUpwindFlux(U[idx], U[idxE], 1.0f, 0.0f, 0.0f);

    residual[idx].rho += fluxE.mass * areaX / volume;
    residual[idx].rhoU += fluxE.momX * areaX / volume;
    residual[idx].rhoV += fluxE.momY * areaX / volume;
    residual[idx].rhoW += fluxE.momZ * areaX / volume;
    residual[idx].rhoE += fluxE.energy * areaX / volume;

    // y-direction faces (South and North)
    int idxS = i + (j - 1) * nx + k * nx * ny;
    int idxN = i + (j + 1) * nx + k * nx * ny;

    faceFlux fluxS = computeUpwindFlux(U[idxS], U[idx], 0.0f, 1.0f, 0.0f);
    float areaY = dx * dz;

    residual[idx].rho -= fluxS.mass * areaY / volume;
    residual[idx].rhoU -= fluxS.momX * areaY / volume;
    residual[idx].rhoV -= fluxS.momY * areaY / volume;
    residual[idx].rhoW -= fluxS.momZ * areaY / volume;
    residual[idx].rhoE -= fluxS.energy * areaY / volume;

    faceFlux fluxN = computeUpwindFlux(U[idx], U[idxN], 0.0f, 1.0f, 0.0f);

    residual[idx].rho += fluxN.mass * areaY / volume;
    residual[idx].rhoU += fluxN.momX * areaY / volume;
    residual[idx].rhoV += fluxN.momY * areaY / volume;
    residual[idx].rhoW += fluxN.momZ * areaY / volume;
    residual[idx].rhoE += fluxN.energy * areaY / volume;

    // z-direction faces (Bottom and Top)
    int idxB = i + j * nx + (k - 1) * nx * ny;
    int idxT = i + j * nx + (k + 1) * nx * ny;

    faceFlux fluxB = computeUpwindFlux(U[idxB], U[idx], 0.0f, 0.0f, 1.0f);
    float areaZ = dx * dy;

    residual[idx].rho -= fluxB.mass * areaZ / volume;
    residual[idx].rhoU -= fluxB.momX * areaZ / volume;
    residual[idx].rhoV -= fluxB.momY * areaZ / volume;
    residual[idx].rhoW -= fluxB.momZ * areaZ / volume;
    residual[idx].rhoE -= fluxB.energy * areaZ / volume;

    faceFlux fluxT = computeUpwindFlux(U[idx], U[idxT], 0.0f, 0.0f, 1.0f);

    residual[idx].rho += fluxT.mass * areaZ / volume;
    residual[idx].rhoU += fluxT.momX * areaZ / volume;
    residual[idx].rhoV += fluxT.momY * areaZ / volume;
    residual[idx].rhoW += fluxT.momZ * areaZ / volume;
    residual[idx].rhoE += fluxT.energy * areaZ / volume;
}

__global__ void solverKernels::applyBoundaryConditions
(
    conservativeVars* U,
    int nx,
    int ny,
    int nz
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Apply no-slip boundary conditions on all 6 faces
    for (int idx = tid; idx < nx * ny * nz; idx += totalThreads)
    {
        int i = idx % nx;
        int j = (idx / nx) % ny;
        int k = idx / (nx * ny);

        if (i == 0 || i == nx - 1)
        {
            U[idx].rhoU = 0.0f;
        }

        if (j == 0 || j == ny - 1)
        {
            U[idx].rhoV = 0.0f;
        }

        if (k == 0 || k == nz - 1)
        {
            U[idx].rhoW = 0.0f;
        }
    }
}

__global__ void solverKernels::integrateTime
(
    conservativeVars* U,
    conservativeVars* residual,
    float dt,
    int nx,
    int ny,
    int nz
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if
    (
        i < 1 || i >= nx - 1
     || j < 1 || j >= ny - 1
     || k < 1 || k >= nz - 1
    ) return;

    int idx = i + j * nx + k * nx * ny;

    // Forward Euler update: U^(n+1) = U^n + dt * R(U^n)
    U[idx].rho += dt * residual[idx].rho;
    U[idx].rhoU += dt * residual[idx].rhoU;
    U[idx].rhoV += dt * residual[idx].rhoV;
    U[idx].rhoW += dt * residual[idx].rhoW;
    U[idx].rhoE += dt * residual[idx].rhoE;

    // Clip density to ensure physical values
    U[idx].rho = fmaxf(U[idx].rho, 1e-6f);
    U[idx].rhoE = fmaxf(U[idx].rhoE, 1e-6f);
}