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
#include <fstream>
#include <set>
#include <unordered_map>

#include "cartMesh.h"

std::string cartMesh::meshFile_ = "meshInput.txt";

void cartMesh::createMesh()
{
    std::string inputFile = "./" + meshFile_;
    std::cout << "Reading mesh from input file: " << inputFile << std::endl;

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

cartMesh::cartMesh()
:
    scale_(1.0),
    blockX_(-1),
    blockY_(-1),
    blockZ_(-1),
    lengthX_(-1.0),
    lengthY_(-1.0),
    lengthZ_(-1.0)
{
    createMesh();
    checkMesh();
}

bool cartMesh::checkMesh() const
{
    if (scale_ < 0.0)
    {
        std::cerr << "Mesh scale is invalid: " << scale_ << std::endl;
        return false;
    }

    if (blockX_ < 0 || blockY_ < 0 || blockZ_ < 0)
    {
        std::cerr << "Number of mesh elements is invalid: "
            << blockX_ << " " << blockY_ << " " << blockZ_ << std::endl;
        return false;
    }

    if (lengthX_ < 0.0 || lengthY_ < 0.0 || lengthZ_ < 0.0)
    {
        std::cerr << "Length of mesh elements is invalid: "
            << lengthX_ << " " << lengthY_ << " " << lengthZ_ << std::endl;
        return false;
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

    // ...

    return true;
}