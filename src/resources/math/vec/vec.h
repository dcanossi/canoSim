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

    // Stream operator
    friend std::ostream& operator<<(std::ostream& os, const vec& vector);
};

#endif