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

#ifndef vtkWriter_H
#define vtkWriter_H

#include "face.h"
#include "vec.h"

class vtkWriter
{
    std::vector<vec> points_;

    std::vector<face> faces_;

    std::string fileName_;

public:

    vtkWriter
    (
        const std::vector<vec>& points,
        const std::vector<face>& faces,
        const std::string& fileName
    );

    void write() const;
};

#endif