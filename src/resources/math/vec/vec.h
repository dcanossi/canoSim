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

#ifndef vec_H
#define vec_H

#include <iostream>

class vec
{
    float vecX_, vecY_, vecZ_;

public:

    // Default construction
    vec() = default;

    // Construction from components
    vec(const float& x, const float& y, const float& z);

    // Get x-component
    const float& x()
    {
        return vecX_;
    }

    // Get y-component
    const float& y()
    {
        return vecY_;
    }

    // Get x-component
    const float& z()
    {
        return vecZ_;
    }

    // i-element access operator
    inline const float& operator[](int direction) const;

    // i-element non-const access operator
    inline float& operator[](int direction);

    // Stream operator
    friend std::ostream& operator<<(std::ostream& os, const vec& vector);
};

// Inline functions definition
inline const float& vec::operator[](int direction) const
{
    switch (direction)
    {
        case 0:
            return vecX_;
        case 1:
            return vecY_;
        case 2:
            return vecZ_;
        default:
            std::cerr << "Error: Index out of range for vec" << std::endl;
            std::abort();
    }
}

inline float& vec::operator[](int direction)
{
    switch (direction)
    {
        case 0:
            return vecX_;
        case 1:
            return vecY_;
        case 2:
            return vecZ_;
        default:
            std::cerr << "Error: Index out of range for vec" << std::endl;
            std::abort();
    }
}

#endif