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

#include <fstream>

#include "vtkWriter.h"

vtkWriter::vtkWriter
(
    const std::vector<vec>& points,
    const std::vector<face>& faces,
    const std::string& fileName
)
:
    points_(points),
    faces_(faces),
    fileName_(fileName)
{}

void vtkWriter::write() const
{
    std::filebuf fb;
    fb.open(fileName_, std::ios::out);
    std::ostream outStr(&fb);

    // Write VTK header
    outStr << "# vtk DataFile Version 4.0\n";
    outStr << "vtk output\n";
    outStr << "ASCII\n";
    outStr << "DATASET POLYDATA\n";
    outStr << "POINTS " << points_.size() << " float\n";

    // Write all points
    for (const vec& pointi : points_)
    {
        outStr << pointi[0] << " " << pointi[1] << " " << pointi[2] << "\n";
    }

    outStr << "POLYGONS ";

    int nFaceNodes = 0;
    for (const face& facei : faces_)
    {
        nFaceNodes += facei.size();
    }

    outStr << faces_.size() << " " << faces_.size() + nFaceNodes << "\n";

    // Write all faces
    for (const face& facei : faces_)
    {
        outStr << facei.size();
        for (auto pointi : facei)
        {
            outStr << " " << pointi;
        }
        outStr << "\n";
    }
}