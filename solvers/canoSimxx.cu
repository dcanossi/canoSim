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

#include "gFunctions.h"
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

int main(int argc, char** argv)
{
    std::unordered_set<std::string> runOpt = {"-mesh", "-init", "-solve"};

    // Parse run option
    if (argc != 2 || !runOpt.count(argv[1]))
    {
        std::cout << "Invalid usage!\nPlease provide one of the following "
            << "run options:\n"
            << "  -mesh\n"
            << "  -init\n"
            << "  -solve\n";
        std::exit(1);
    }

    printHeader();
    printRunInfo();

    std::string runFlag = argv[1];
    runFlag.erase(0, 1);
    if (runFlag == "mesh")
    {
        std::cout << "Creating mesh for simulation..." << std::endl;
        const cartMesh mesh;

        std::cout << "\nEnd." << std::endl;
    }
    else if (runFlag == "init")
    {
        std::cout << "Initializing fields\n" << std::endl;

        std::cout << "End." << std::endl;
    }
    else if (runFlag == "solve")
    {
        std::cout << "Starting simulation\n" << std::endl;

        std::cout << "End." << std::endl;
    }

    return 0;
}