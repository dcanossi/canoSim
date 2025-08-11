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

__host__ __device__ inline primitiveVars solverKernels::conservativeToPrimitive
(
    const conservativeVars& cons
)
{
    primitiveVars prim;
    prim.rho = cons.rho;

    // Avoid division by zero
    if (cons.rho < 1e-10f)
    {
        prim.u = prim.v = prim.w = prim.p = 0.0f;
        return prim;
    }

    prim.u = cons.rhoU / cons.rho;
    prim.v = cons.rhoV / cons.rho;
    prim.w = cons.rhoW / cons.rho;

    float kinetic =
        0.5f * cons.rho * (prim.u * prim.u + prim.v * prim.v + prim.w * prim.w);

    prim.p = (GAMMA - 1.0f) * (cons.rhoE - kinetic);

    // Ensure positive pressure
    prim.p = fmaxf(prim.p, 1e-6f);

    return prim;
}

__host__ __device__
inline conservativeVars solverKernels::primitiveToConservative
(
    const primitiveVars& prim
)
{
    conservativeVars cons;
    cons.rho = prim.rho;
    cons.rhoU = prim.rho * prim.u;
    cons.rhoV = prim.rho * prim.v;
    cons.rhoW = prim.rho * prim.w;

    float kinetic =
        0.5f * prim.rho * (prim.u * prim.u + prim.v * prim.v + prim.w * prim.w);

    cons.rhoE = prim.p / (GAMMA - 1.0f) + kinetic;

    return cons;
}

__device__ inline faceFlux solverKernels::computeUpwindFlux
(
    const conservativeVars& consL,
    const conservativeVars& consR,
    float nx,
    float ny,
    float nz
)
{
    faceFlux flux;

    // Convert to primitive variables
    primitiveVars primL = conservativeToPrimitive(consL);
    primitiveVars primR = conservativeToPrimitive(consR);

    // Compute normal velocities
    float unL = primL.u * nx + primL.v * ny + primL.w * nz;
    float unR = primR.u * nx + primR.v * ny + primR.w * nz;

    // Simple upwind scheme based on normal velocity
    float un = 0.5f * (unL + unR);

    if (un > 0)
    {
        flux.mass = consL.rho * unL;
        flux.momX = consL.rhoU * unL + primL.p * nx;
        flux.momY = consL.rhoV * unL + primL.p * ny;
        flux.momZ = consL.rhoW * unL + primL.p * nz;
        flux.energy = (consL.rhoE + primL.p) * unL;
    }
    else
    {
        flux.mass = consR.rho * unR;
        flux.momX = consR.rhoU * unR + primR.p * nx;
        flux.momY = consR.rhoV * unR + primR.p * ny;
        flux.momZ = consR.rhoW * unR + primR.p * nz;
        flux.energy = (consR.rhoE + primR.p) * unR;
    }

    return flux;
}