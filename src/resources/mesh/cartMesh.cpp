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

#include <algorithm>
#include <cfloat>
#include <fstream>
#include <set>
#include <unordered_map>

#include "cartMesh.h"

// #define DEBUG;

#ifdef DEBUG
#define DEBUG_COUT(str) do { std::cout << str << std::endl; } while (false)
#else
#define DEBUG_COUT(str) do { } while (false)
#endif

std::string cartMesh::meshFile_ = "meshInput.txt";

void cartMesh::readMeshInput()
{
    std::string inputFile = "./" + meshFile_;
    std::cout << "Reading mesh from input file: " << inputFile << std::endl;

    readOrigin();

    // Lambda for removing specific delimiters from input
    auto isDelim = [](const char& chari)
    {
        switch (chari)
        {
            case ' ':
                return true;
            case ';':
                return true;
            default:
                return false;
        }
    };

    // List of all input keywords from the mesh file
    std::set<std::string> inputKeys
        {
            "scale",
            "nx",
            "ny",
            "nz",
            "lx",
            "ly",
            "lz"
        };

    const size_t inputSize = inputKeys.size();
    std::unordered_map<std::string, float> keyFound;

    std::ifstream file(meshFile_);
    std::string str;
    while (std::getline(file, str))
    {
        for (auto key : inputKeys)
        {
            if (keyFound.find(key) != keyFound.end()) continue;

            if (str.find(key) != std::string::npos)
            {
                std::string value(str.substr(str.find(key) + key.length()));

                // Remove delimiters from input
                value.erase
                (
                    std::remove_if(value.begin(), value.end(), isDelim),
                    value.end()
                );

                keyFound.insert({key, std::atof(value.c_str())});
            }
        }
    }

    // Sanity check for missing inputs
    if (keyFound.size() != inputSize)
    {
        std::cerr << "Invalid mesh input." << std::endl;
    }

    // Construct mesh data from stream
    scale_ = keyFound["scale"];

    blockX_ = static_cast<int>(keyFound["nx"]);
    blockY_ = static_cast<int>(keyFound["ny"]);
    blockZ_ = static_cast<int>(keyFound["nz"]);

    lengthX_ = keyFound["lx"];
    lengthY_ = keyFound["ly"];
    lengthZ_ = keyFound["lz"];
}

void cartMesh::generateCellAddr
(
    int i,
    int j,
    int k,
    int pIdx,
    std::vector<std::set<int>>& addr
) const
{
    unsigned int idx_x = std::max(i - 1, 0);
    unsigned int idx_y = std::max(j - 1, 0);
    unsigned int idx_z = std::max(k - 1, 0);

    unsigned int idx = idx_x + idx_y * blockX_ + idx_z * blockX_ * blockY_;

    DEBUG_COUT("\nInserting pIndex " << pIdx
        << " (" << points_[pIdx] << ") in cell " << idx);

    addr[idx].insert(pIdx);

    bool nextInX = false;
    bool nextInY = false;
    bool nextInZ = false;

    // Insert vertice into the next x-direction cell
    if (i != 0 && i < blockX_)
    {
        const int nextIdx =
            (idx_x + 1) + idx_y * blockX_ + idx_z * blockX_ * blockY_;

        DEBUG_COUT("Inserting pIndex " << pIdx << " (" << points_[pIdx]
            << ") in (next-x) cell " << nextIdx);

        addr[nextIdx].insert(pIdx);

        nextInX = true;
    }

    // Insert vertice into the next y-direction cell
    if (j != 0 && j < blockY_)
    {
        const int nextIdx =
            idx_x + (idx_y + 1) * blockX_ + idx_z * blockX_ * blockY_;

        DEBUG_COUT("Inserting pIndex " << pIdx << " (" << points_[pIdx]
            << ") in (next-y) cell " << nextIdx);

        addr[nextIdx].insert(pIdx);

        nextInY = true;
    }

    // Insert vertice into the next z-direction cell
    if (k != 0 && k < blockZ_)
    {
        const int nextIdx =
            idx_x + idx_y * blockX_ + (idx_z + 1) * blockX_ * blockY_;

        DEBUG_COUT("Inserting pIndex " << pIdx << " (" << points_[pIdx]
            << ") in (next-z) cell " << nextIdx);

        addr[nextIdx].insert(pIdx);

        nextInZ = true;
    }

    // Insert additional vertices if previously added to x, y and/or z
    if (nextInX && nextInY)
    {
        const int nextIdx =
            (idx_x + 1) + (idx_y + 1) * blockX_ + idx_z * blockX_ * blockY_;

        DEBUG_COUT("Inserting pIndex " << pIdx << " (" << points_[pIdx]
            << ") in (next-xy) cell " << nextIdx);

        addr[nextIdx].insert(pIdx);
    }
    if (nextInX && nextInZ)
    {
        const int nextIdx =
            (idx_x + 1) + idx_y * blockX_ + (idx_z + 1) * blockX_ * blockY_;

        DEBUG_COUT("Inserting pIndex " << pIdx << " (" << points_[pIdx]
            << ") in (next-xz) cell " << nextIdx);

        addr[nextIdx].insert(pIdx);
    }
    if (nextInY && nextInZ)
    {
        const int nextIdx =
            idx_x + (idx_y + 1) * blockX_ + (idx_z + 1) * blockX_ * blockY_;

        DEBUG_COUT("Inserting pIndex " << pIdx << " (" << points_[pIdx]
            << ") in (next-yz) cell " << nextIdx);

        addr[nextIdx].insert(pIdx);
    }
    if (nextInX && nextInY && nextInZ)
    {
        const int nextIdx =
            (idx_x + 1)
          + (idx_y + 1) * blockX_
          + (idx_z + 1) * blockX_ * blockY_;

        DEBUG_COUT("Inserting pIndex " << pIdx << " (" << points_[pIdx]
            << ") in (next-xyz) cell " << nextIdx);

        addr[nextIdx].insert(pIdx);
    }
}

bool cartMesh::createMesh()
{
    // Generate cell list
    const int nCells = blockX_ * blockY_ * blockZ_;
    cells_.reserve(nCells);

    // Generate point list
    points_.reserve((blockX_ + 1) * blockY_ * (blockZ_ + 1));

    std::vector<std::set<int>> cellsVertices;
    cellsVertices.resize(nCells);

    int pIdx = 0;
    for (int k = 0; k < blockZ_ + 1; k++)
    {
        for (int j = 0; j < blockY_ + 1; j++)
        {
            for (int i = 0; i < blockX_ + 1; i++)
            {
                points_.emplace_back(vec(i*lengthX_, j*lengthY_, k*lengthZ_));

                // Generate cell-point addressing
                generateCellAddr(i, j, k, pIdx, cellsVertices);

                pIdx++;
            }
        }
    }

    #ifdef DEBUG
    {
        std::cout << "\nPoint coordinates:\n" << std::endl;
        for (size_t i = 0; i < points_.size(); i++)
        {
            std::cout << "Point " << i << ": " << points_[i] << std::endl;
        }

        std::cout << "\nCell-point addressing:" << std::endl;
        for (size_t i = 0; i < cellsVertices.size(); i++)
        {
            std::cout << std::endl;
            for (auto j : cellsVertices[i])
            {
                std::cout << "Cell " << i << ": " << j << std::endl;
            }
        }
    }
    #endif

    // Sanity check for the number of vertices in each cell
    for (size_t i = 0; i < cellsVertices.size(); i++)
    {
        if (cellsVertices[i].size() != nCellPoints_)
        {
            std::cerr << "\nError: Invalid number of points in cell " << i
                << ": " << cellsVertices[i].size() << std::endl;

            std::abort();
        }
    }

    // Generate cells with point addressing
    for (int i = 0; i < nCells; i++)
    {
        cells_.emplace_back(cellsVertices[i], points_);
    }

    return true;
}

cartMesh::cartMesh()
:
    origin_(0, 0, 0),
    scale_(1.0),
    blockX_(-1),
    blockY_(-1),
    blockZ_(-1),
    lengthX_(-1.0),
    lengthY_(-1.0),
    lengthZ_(-1.0),
    cells_()
{
    // Read mesh input from file and check consistency
    readMeshInput();
    checkMesh();

    // Generate mesh data (points, cells, connectivity)
    createMesh();

    // Write mesh to disk
    write();
}

bool cartMesh::checkMesh() const
{
    if (scale_ < 0.0)
    {
        std::cerr << "\nError: Mesh scale is invalid: " << scale_ << std::endl;

        std::abort();
    }

    if (blockX_ <= 0 || blockY_ <= 0 || blockZ_ <= 0)
    {
        std::cerr << "\nError: Number of mesh elements is invalid: "
            << blockX_ << " " << blockY_ << " " << blockZ_ << std::endl;

        std::abort();
    }

    if (lengthX_ < FLT_MIN || lengthY_ < FLT_MIN || lengthZ_ < FLT_MIN)
    {
        std::cerr << "\nError: Length of mesh elements is invalid: "
            << lengthX_ << " " << lengthY_ << " " << lengthZ_ << std::endl;

        std::abort();
    }

    return true;
}

void cartMesh::printMeshStats() const
{
    std::cout << "\n------------------"
        << "\n    Mesh stats"
        << "\n------------------" << std::endl;

    std::cout << "\nMesh cells:\n"
        << "    Nx = " << blockX_
        << "\n    Ny = " << blockY_
        << "\n    Nz = " << blockZ_ << std::endl;

    std::cout << "Elements length:\n"
        << "    Lx = " << lengthX_
        << "\n    Ly = " << lengthY_
        << "\n    Lz = " << lengthZ_ << std::endl;

    std::cout << "Mesh scale = " << scale_ << std::endl;
}

bool cartMesh::write() const
{
    std::cout << "\nWriting mesh to disk." << std::endl;

    // To do: Write mesh in vtk format

    return true;
}