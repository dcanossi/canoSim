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
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <unordered_map>

#include "gFunctions.h"
#include "cartMesh.h"
#include "vtkWriter.h"

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

void cartMesh::calcCellAddr
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

std::vector<face> cartMesh::generateFaces
(
    const std::vector<std::set<int>>& vertices,
    std::vector<std::array<std::vector<int>, 6>>& cellsFaces
)
{
    std::vector<face> globalFaces;

    for (size_t i = 0; i < vertices.size(); i++)
    {
        const auto& celli = vertices[i];

        const int minPointIdx = *celli.begin();
        const int maxPointIdx = *celli.rbegin();

        auto& cellFaces = cellsFaces[i];

        // For each cell, calculate the point addressing of each face.
        for (const auto& verti : celli)
        {
            // Insert points in faces perpendicular to x
            if (abs(points_[verti].x() - points_[minPointIdx].x()) < FLT_MIN)
            {
                cellFaces[0].push_back(verti);
            }
            else if
            (
                abs(points_[verti].x() - points_[maxPointIdx].x()) < FLT_MIN
            )
            {
                cellFaces[1].push_back(verti);
            }

            // Insert points in faces perpendicular to y
            if (abs(points_[verti].y() - points_[minPointIdx].y()) < FLT_MIN)
            {
                cellFaces[2].push_back(verti);
            }
            else if
            (
                abs(points_[verti].y() - points_[maxPointIdx].y()) < FLT_MIN
            )
            {
                cellFaces[3].push_back(verti);
            }

            // Insert points in faces perpendicular to z
            if (abs(points_[verti].z() - points_[minPointIdx].z()) < FLT_MIN)
            {
                cellFaces[4].push_back(verti);
            }
            else if
            (
                abs(points_[verti].z() - points_[maxPointIdx].z()) < FLT_MIN
            )
            {
                cellFaces[5].push_back(verti);
            }
        }

        const auto& prevCellFaces =
            i > 0
          ? cellsFaces[i - 1]
          : std::array<std::vector<int>, 6>{};

        for (size_t j = 0; j < cellFaces.size(); j++)
        {
            auto& cellFacej = cellFaces[j];

            // First, check if face has the correct number of points.
            if (cellFacej.size() != nFacePoints_)
            {
                std::cerr << "\nError: Invalid number of points in face " << j
                    << ": " << faces_[j].size() << std::endl;

                std::abort();
            }

            // Swap order of last two face vertices, so it forms a topologically
            // closed domain.
            std::swap
            (
                cellFacej[cellFacej.size() - 2],
                cellFacej[cellFacej.size() - 1]
            );

            // Then, check if this face is already included in the global faces.
            // If so, then mark it as an internal face and do not add it again.
            std::vector<int> prevCellNextFacej;
            if (!prevCellFaces[0].empty() && j < cellFaces.size() - 1)
            {
                prevCellNextFacej = prevCellFaces[j + 1];
            }

            if (cellFacej == prevCellNextFacej)
            {
                #ifdef DEBUG
                {
                    std::cout << "\nNot including duplicated face " << j
                        << " from cell " << i << std::endl;
                }
                #endif

                for (int k = globalFaces.size() - 1; k >= 0; --k)
                {
                    if (globalFaces[k] == cellFacej)
                    {
                        globalFaces[k].isBoundary() = false;
                    }
                }

                continue;
            }

            // Finally, add the new face to the global face list.
            globalFaces.emplace_back(face(cellFacej));
        }
    }

    #ifdef DEBUG
    {
        std::cout << "\nFace-point addressing:\n" << std::endl;
        int cnt = 0;
        for (const auto& facei : globalFaces)
        {
            std::cout << "Face: " << cnt++ << ": " << facei;

            if (facei.isBoundary())
            {
                std::cout << " | Boundary" << std::endl;
            }
            else
            {
                std::cout << " | Internal" << std::endl;
            }
        }
    }
    #endif

    // Sanity check for the number of vertices in each face
    for (size_t i = 0; i < globalFaces.size(); i++)
    {
        if (globalFaces[i].size() != nFacePoints_)
        {
            std::cerr << "\nError: Invalid number of points in face " << i
                << ": " << faces_[i].size() << std::endl;

            std::abort();
        }
    }

    return globalFaces;
}

bool cartMesh::createMesh()
{
    // Generate point list
    points_.reserve((blockX_ + 1) * blockY_ * (blockZ_ + 1));

    // Generate cell list
    const int nCells = blockX_ * blockY_ * blockZ_;
    cells_.reserve(nCells);

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
                calcCellAddr(i, j, k, pIdx, cellsVertices);

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

    // Generate global faces from cells
    std::vector<std::array<std::vector<int>, 6>> cellsFaces;
    cellsFaces.resize(cellsVertices.size());

    faces_ = generateFaces(cellsVertices, cellsFaces);

    // Generate cells with point and face addressing
    for (int i = 0; i < nCells; i++)
    {
        cells_.emplace_back(cellsVertices[i], cellsFaces[i], points_);
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

    // Print mesh statistics
    printMeshStats();

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

    std::filesystem::create_directory("mesh");

    std::unique_ptr<vtkWriter> vtkW =
        std::make_unique<vtkWriter>(points_, faces_, "mesh/cartMesh.vtk");

    vtkW->write();

    return true;
}