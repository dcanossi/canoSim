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

#include <iomanip>
#include <sstream>

#include "clock.h"

const char* clock::monthNames[] =
{
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
};

std::string clock::date()
{
    std::ostringstream osBuf;

    time_t t = time(reinterpret_cast<time_t*>(0));
    struct tm* timeStruct = localtime(&t);

    osBuf << timeStruct->tm_mday << '-' << std::setw(2) << std::setfill('0')
        << monthNames[timeStruct->tm_mon]
        << '-' << std::setw(4) << timeStruct->tm_year + 1900;

    return osBuf.str();
}

std::string clock::clockTime()
{
    std::ostringstream osBuf;

    time_t t = time(reinterpret_cast<time_t*>(0));
    struct tm *timeStruct = localtime(&t);

    osBuf << std::setfill('0')
        << std::setw(2) << timeStruct->tm_hour
        << ':' << std::setw(2) << timeStruct->tm_min
        << ':' << std::setw(2) << timeStruct->tm_sec;

    return osBuf.str();
}