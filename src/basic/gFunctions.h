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

#ifndef gFunctions_H
#define gFunctions_H

#include <bits/stdc++.h>

// --- Global functions definition --- //

// Global debug flag
// #define DEBUG 1;

#ifdef DEBUG
#define DEBUG_COUT(str) do { std::cout << str << std::endl; } while (false)
#else
#define DEBUG_COUT(str) do { } while (false)
#endif

// Overload stream operator << for std::vector
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector)
{
    for (auto i : vector)
    {
        os << i << " ";
    }

    return os;
}

#endif