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

#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>

#include "clock.h"
#include "cartMesh.h"

void printHeader()
{
    std::cout
        << R"(┌──────────────────────────────────────────────────────────────────────────────┐
│       ╔═══╗   ╔═══╗   ╔═══╗   ╔═══╗   ╔═══╗   ╔═══╗   ╔═══╗   ╔══════╗       │
│       ║ c ║   ║ a ║   ║ n ║   ║ o ║   ║ S ║   ║ i ║   ║ m ║   ║ +  + ║       │
│       ╚═══╝   ╚═══╝   ╚═══╝   ╚═══╝   ╚═══╝   ╚═══╝   ╚═══╝   ╚══════╝       │
│                                                                              │
│                       ~ Computational Fluid Dynamics ~                       │
│                       High-Performance GPU Flow Solver                       │
│                                                                              │
│                                                                              │
│                                            Copyright (c) 2025 Dário Canossi  │
└──────────────────────────────────────────────────────────────────────────────┘
)";
}

void printRunInfo()
{
    // Host name
    char hostBuf[32];
    gethostname(hostBuf, sizeof(hostBuf));

    // Device ID and properties
    int devID;
    cudaDeviceProp props;
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&props, devID);

    std::cout << "| PID    : " << getpid()
        << "\n| Host   : " << hostBuf
        << "\n| Device : " << "[" << devID << "] " << props.name << " (CC: "
            << props.major << "." << props.minor << ")"
        << "\n| Date   : " << clock::date().c_str()
        << "\n| Time   : " << clock::clockTime().c_str() << "\n"
        << "########################################"
        << "########################################"
        << "\n" << std::endl;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

int main()
{
    printHeader();
    printRunInfo();

    std::cout << "Starting simulation\n" << std::endl;

    std::cout << "Creating mesh for simulation..." << std::endl;
    const cartMesh mesh;
    mesh.printMeshStats();

    std::cout << "\nEnd." << std::endl;

    return 0;
}