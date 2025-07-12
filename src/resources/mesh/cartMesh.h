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

#ifndef cartMesh_H
#define cartMesh_H

#include <iostream>

class cartMesh
{
    static std::string meshFile_;

    float scale_;

    int blockX_, blockY_, blockZ_;

    float lengthX_, lengthY_, lengthZ_;

    // Helper function for mesh reading
    void createMesh();

public:

    // Construction from mesh file
    cartMesh();

    // Run a mesh check
    bool checkMesh() const;

    // Print mesh stats
    void printMeshStats() const;

    // Write mesh
    bool write() const;
};

#endif