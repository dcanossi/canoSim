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

#ifndef face_H
#define face_H

#include <iostream>
#include <vector>

struct face : std::vector<int>
{
    enum class direction {X, Y, Z};

    bool boundary_;

public:

    // Null construction
    face() = default;

    // Construction from list of vertices
    face(const std::vector<int>& vertices)
    :
        std::vector<int>(vertices),
        boundary_(true)
    {}

    // Is this face a boundary face?
    bool isBoundary() const
    {
        return boundary_;
    }

    // Return the isBoundary status for this face so it can be changed
    bool& isBoundary()
    {
        return boundary_;
    }

    // Stream operator
    friend std::ostream& operator<<(std::ostream& os, const face& face)
    {
        os << "( ";
        for (auto pointi : face)
        {
            os << pointi << " ";
        }
        os << ")";

        return os;
    }
};

#endif