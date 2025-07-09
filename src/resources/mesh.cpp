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

class mesh
{
    int blockX, blockY, blockZ;

public:

    mesh(int nx, int ny, int nz)
    :
        blockX(nx),
        blockY(ny),
        blockZ(nz)
    {
        checkMesh();
    }

    // Run a mesh check
    bool checkMesh() const
    {
        return true;
    }

    // Print mesh stats
    void printMeshStats() const
    {
        std::cout << "Number of blocks: Nx = " << blockX
            << " | Ny = " << blockY
            << " | Nz = " << blockZ << std::endl;
    }
};