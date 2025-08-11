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
#include <filesystem>
#include <fstream>

#include "EulerSolver.h"
#include "solverKernels.h"

std::string EulerSolver::controlFile_ = "controls.txt";

EulerSolver::EulerSolver(cartMesh&& mesh)
:
    mesh_(std::make_unique<cartMesh>(std::forward<cartMesh>(mesh))),
    nx_(mesh_->getBlockX() + 2),
    ny_(mesh_->getBlockY() + 2),
    nz_(mesh_->getBlockZ() + 2),
    dx_(mesh_->getLengthX()),
    dy_(mesh_->getLengthY()),
    dz_(mesh_->getLengthZ()),
    totalCells_(nx_ * ny_ * nz_),
    startTime_(-1),
    nIter_(-1),
    CFL_(-1),
    dt_(0.0f),
    totalTime_(0.0f),
    d_U_(nullptr),
    d_residual_(nullptr),
    h_U_(nullptr)
{
    // Read simulation controls
    readControls();

    // Allocate host memory
    h_U_ = new conservativeVars[totalCells_];

    // Allocate device memory
    cudaMalloc(&d_U_, totalCells_ * sizeof(conservativeVars));
    cudaMalloc(&d_residual_, totalCells_ * sizeof(conservativeVars));

    // Initialise to zero
    cudaMemset(d_U_, 0, totalCells_ * sizeof(conservativeVars));
    cudaMemset(d_residual_, 0, totalCells_ * sizeof(conservativeVars));
}

EulerSolver::~EulerSolver()
{
    delete[] h_U_;
    cudaFree(d_U_);
    cudaFree(d_residual_);
}

void EulerSolver::readControls()
{
    std::string inputFile = "./" + controlFile_;
    std::cout << "\nReading control parameters from input file: "
        << inputFile << std::endl;

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

    // List of all input keywords from the control file
    std::set<std::string> inputKeys
        {
            "startTime",
            "nIter",
            "CFL"
        };

    std::unordered_map<std::string, float> keyFound;

    std::ifstream file(controlFile_);
    std::string str;
    while (std::getline(file, str))
    {
        for (const auto& key : inputKeys)
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
    if (keyFound.size() != inputKeys.size())
    {
        std::cerr << "Error: Invalid simulation control input!" << std::endl;
        std::exit(1);
    }

    // Construct control data from stream
    startTime_ = keyFound["startTime"];
    nIter_ = keyFound["nIter"];
    CFL_ = keyFound["CFL"];
}

void EulerSolver::solve()
{
    totalTime_ += startTime_;
    std::cout << std::endl;

    // Initialise with uniform flow
    initialiseUniform(1.225f, 50.0f, 10.0f, 0.0f, 101325.0f);

    // Run for the prescribed iterations
    for (int iter = 1; iter <= nIter_; ++iter)
    {
        runTimeStep();

        if (iter % 10 == 0)
        {
            std::cout << "Iteration " << iter << " | dt = "
                << getTimeStep() << "s | Time = " << totalTime_ << std::endl;
        }
    }

    writeSolution("finalSolution.vtk", nIter_);
}

void EulerSolver::initialiseUniform
(
    float rho,
    float u,
    float v,
    float w,
    float p
)
{
    primitiveVars prim(rho, u, v, w, p);
    conservativeVars cons = solverKernels::primitiveToConservative(prim);

    for (size_t i = 0; i < totalCells_; ++i)
    {
        h_U_[i] = cons;
    }

    cudaMemcpy
    (
        d_U_,
        h_U_,
        totalCells_ * sizeof(conservativeVars),
        cudaMemcpyHostToDevice
    );
}

void EulerSolver::computeResiduals()
{
    dim3 blockSize(8, 8, 4);
    dim3 gridSize
    (
        (nx_ + blockSize.x - 1) / blockSize.x,
        (ny_ + blockSize.y - 1) / blockSize.y,
        (nz_ + blockSize.z - 1) / blockSize.z
    );

    solverKernels::computeResidual<<<gridSize, blockSize>>>
        (d_U_, d_residual_, nx_, ny_, nz_, dx_, dy_, dz_);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error in computeResiduals: "
            << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}

float EulerSolver::computeTimeStep()
{
    // For now, use a conservative estimate for max speed of wave propagation.
    // In practice, you'd compute this quantity from the solution.
    float maxWaveSpeed = 100.0f;  // m/s

    float dtx = CFL_ * dx_ / maxWaveSpeed;
    float dty = CFL_ * dy_ / maxWaveSpeed;
    float dtz = CFL_ * dz_ / maxWaveSpeed;

    dt_ = std::min({dtx, dty, dtz});

    // Update total simulation time
    totalTime_ += dt_;

    return dt_;
}

void EulerSolver::applyBoundaryConditions()
{
    int blockSize = 256;
    int gridSize = (totalCells_ + blockSize - 1) / blockSize;

    solverKernels::applyBoundaryConditions<<<gridSize, blockSize>>>
        (d_U_, nx_, ny_, nz_);

    cudaDeviceSynchronize();
}

void EulerSolver::updateSolution()
{
    dim3 blockSize(8, 8, 4);
    dim3 gridSize
    (
        (nx_ + blockSize.x - 1) / blockSize.x,
        (ny_ + blockSize.y - 1) / blockSize.y,
        (nz_ + blockSize.z - 1) / blockSize.z
    );

    solverKernels::integrateTime<<<gridSize, blockSize>>>
        (d_U_, d_residual_, dt_, nx_, ny_, nz_);

    cudaDeviceSynchronize();
}

void EulerSolver::runTimeStep()
{
    computeTimeStep();
    computeResiduals();
    updateSolution();
    applyBoundaryConditions();
}

void EulerSolver::getSolution(conservativeVars* hostData)
{
    cudaMemcpy
    (
        hostData,
        d_U_,
        totalCells_ * sizeof(conservativeVars),
        cudaMemcpyDeviceToHost
    );
}

void EulerSolver::writeSolution(const std::string& filename, int iteration)
{
    getSolution(h_U_);

    // Convert to primitive variables for output
    std::vector<float> rho(totalCells_);
    std::vector<float> u(totalCells_);
    std::vector<float> v(totalCells_);
    std::vector<float> w(totalCells_);
    std::vector<float> p(totalCells_);

    for (size_t i = 0; i < totalCells_; ++i)
    {
        primitiveVars prim = solverKernels::conservativeToPrimitive(h_U_[i]);

        rho[i] = prim.rho;
        u[i] = prim.u;
        v[i] = prim.v;
        w[i] = prim.w;
        p[i] = prim.p;
    }

    // Write solution fields to a VTK output file
    std::filesystem::create_directory("solution");
    std::ofstream file("solution/" + filename);

    file << "# vtk DataFile Version 3.0\n";
    file << "Euler Solution at iteration " << iteration << "\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << nx_ << " " << ny_ << " " << nz_ << "\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING " << dx_ << " " << dy_ << " " << dz_ << "\n";
    file << "POINT_DATA " << totalCells_ << "\n";

    file << "SCALARS density float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (const auto& val : rho) file << val << "\n";

    file << "SCALARS pressure float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (const auto& val : p) file << val << "\n";

    file << "VECTORS velocity float\n";
    for (size_t i = 0; i < totalCells_; ++i)
    {
        file << u[i] << " " << v[i] << " " << w[i] << "\n";
    }

    file.close();

    std::cout << "\nFields solution written to " << filename << std::endl;
}