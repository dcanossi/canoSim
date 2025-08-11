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

#include <array>
#include <iostream>
#include <vector>

#include "cell.h"

class cartMesh
{
    static std::string meshFile_;

    vec origin_;

    float scale_;

    int blockX_, blockY_, blockZ_;

    float lengthX_, lengthY_, lengthZ_;

    // nPoints in each cell (8 for a Cartesian mesh)
    static constexpr int nCellPoints_ = 8;

    // nPoints in each face (4 for a Cartesian mesh)
    static constexpr int nFacePoints_ = 4;

    // Mesh points and cells
    std::vector<vec> points_;
    std::vector<cell> cells_;
    std::vector<face> faces_;

    // Helper function for reading the origin from the mesh input
    void readOrigin()
    {
        // Not yet supported. Origin = (0, 0, 0) by default.
    }

    // Helper function for mesh reading
    void readMeshInput();

    // Generate cell-point addressing
    void calcCellAddr
    (
        int i,
        int j,
        int k,
        int pIdx,
        std::vector<std::set<int>>& addr
    ) const;

    // Create global faces with face-point addressing
    std::vector<face> generateFaces
    (
        const std::vector<std::set<int>>& vertices,
        std::vector<std::array<std::vector<int>, 6>>& cellsFaces
    );

    // Helper function for creating mesh
    bool createMesh();

public:

    // Construction from mesh file
    cartMesh();

    // Static function to read mesh
    // static cartMesh readMesh();

    // Run a mesh check
    bool checkMesh() const;

    // Print mesh stats
    void printMeshStats() const;

    // Write mesh
    bool write() const;

    // Getters for mesh dimensions
    int getBlockX() const
    {
        return blockX_;
    }

    int getBlockY() const
    {
        return blockY_;
    }

    int getBlockZ() const
    {
        return blockZ_;
    }

    // Getters for cell dimensions
    float getLengthX() const
    {
        return lengthX_;
    }

    float getLengthY() const
    {
        return lengthY_;
    }

    float getLengthZ() const
    {
        return lengthZ_;
    }
};

#endif