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

#ifndef EulerSolver_H
#define EulerSolver_H

#include <memory>

#include "cartMesh.h"

// Struct for conservative variables (stored per cell)
struct conservativeVars
{
    float rho;
    float rhoU;
    float rhoV;
    float rhoW;
    float rhoE;

    __host__ __device__ conservativeVars()
    :
        rho(0),
        rhoU(0),
        rhoV(0),
        rhoW(0),
        rhoE(0)
    {}

    __host__ __device__ conservativeVars
    (
        float r,
        float ru,
        float rv,
        float rw,
        float re
    )
    :
        rho(r),
        rhoU(ru),
        rhoV(rv),
        rhoW(rw),
        rhoE(re)
    {}
};

// Struct for primitive variables (stored per cell)
struct primitiveVars
{
    float rho;
    float u;
    float v;
    float w;
    float p;

    __host__ __device__ primitiveVars()
    :
        rho(0),
        u(0),
        v(0),
        w(0),
        p(0)
    {}

    __host__ __device__ primitiveVars
    (
        float r,
        float vx,
        float vy,
        float vz,
        float pr
    )
    :
        rho(r),
        u(vx),
        v(vy),
        w(vz),
        p(pr)
    {}
};

// Struct for face fluxes
struct faceFlux
{
    float mass;
    float momX;
    float momY;
    float momZ;
    float energy;

    __host__ __device__ faceFlux()
    :
        mass(0),
        momX(0),
        momY(0),
        momZ(0),
        energy(0)
    {}
};

class EulerSolver
{
    // Name of the solution control file
    static std::string controlFile_;

    // Pointer to underlying mesh object
    std::unique_ptr<cartMesh> mesh_;

    // Grid dimensions (including ghost cells for the boundaries)
    int nx_, ny_, nz_;

    // Cell dimensions
    float dx_, dy_, dz_;

    size_t totalCells_;

    // Time step control
    float startTime_;
    int nIter_;
    float CFL_;
    float dt_;
    float totalTime_;

    // Device memory for solution and residuals
    conservativeVars* d_U_;
    conservativeVars* d_residual_;

    // Host memory for solution
    conservativeVars* h_U_;

public:

    // Construct from mesh
    EulerSolver(cartMesh&& mesh);

    // Destructor
    ~EulerSolver();

    // Read simulation control parameters from input file
    void readControls();

    // Main solve function
    void solve();

    // Initialise flow field with uniform conditions (for testing purposes)
    void initialiseUniform(float rho, float u, float v, float w, float p);

    // Compute residuals on device
    void computeResiduals();

    // Compute stable time step, following the CFL condition.
    float computeTimeStep();

    // Apply boundary conditions
    void applyBoundaryConditions();

    // Update solution with time integration
    void updateSolution();

    // Perform operations within a single time step
    void runTimeStep();

    // Get solution back to host
    void getSolution(conservativeVars* hostData);

    // Write fields solution to VTK file
    void writeSolution(const std::string& filename, int iteration = 0);

    // Getters
    float getTimeStep() const
    {
        return dt_;
    }

    int getNx() const
    {
        return nx_ - 2;
    }

    int getNy() const
    {
        return ny_ - 2;
    }

    int getNz() const
    {
        return nz_ - 2;
    }

    const cartMesh* getMesh() const
    {
        return mesh_.get();
    }
};

#endif