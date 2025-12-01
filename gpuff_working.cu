/**
 * GPUFF-RCAPv3 - Working Version
 * Simplified single-file build for testing
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

// Constants
#define MAX_NUCLIDES 80
#define MAX_ORGANS 20
#define DATA_FIELDS 5
#define PI 3.141592f

// Simple structures
struct NuclideData {
    float halfLife;
    float activity;
    float exposure_data[MAX_ORGANS * DATA_FIELDS];
};

struct SimulationControl {
    float timeStep;
    float totalTime;
    int outputFreq;
};

struct PuffData {
    float x, y, z;          // Position
    float vx, vy, vz;       // Velocity
    float sigma_x, sigma_y, sigma_z;  // Dispersion parameters
    float mass;             // Mass of material
    float time;             // Age of puff
    bool active;            // Is puff active
};

// Device global variables
__device__ float* d_exposure;
float exposure_data_all[MAX_NUCLIDES * MAX_ORGANS * DATA_FIELDS];

// Simple CUDA kernel for puff transport
__global__ void transportPuffs(PuffData* puffs, int numPuffs, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPuffs && puffs[idx].active) {
        // Simple advection
        puffs[idx].x += puffs[idx].vx * dt;
        puffs[idx].y += puffs[idx].vy * dt;
        puffs[idx].z += puffs[idx].vz * dt;

        // Simple dispersion growth
        float growthRate = 0.1f;
        puffs[idx].sigma_x += growthRate * dt;
        puffs[idx].sigma_y += growthRate * dt;
        puffs[idx].sigma_z += growthRate * dt * 0.5f;

        // Update puff age
        puffs[idx].time += dt;
    }
}

// Simple concentration calculation kernel
__global__ void calculateConcentration(
    PuffData* puffs, int numPuffs,
    float* grid, int gridX, int gridY,
    float cellSize, float originX, float originY
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < gridX && iy < gridY) {
        float x = originX + ix * cellSize;
        float y = originY + iy * cellSize;
        float concentration = 0.0f;

        for (int p = 0; p < numPuffs; p++) {
            if (puffs[p].active) {
                float dx = x - puffs[p].x;
                float dy = y - puffs[p].y;

                // Gaussian puff formula (simplified)
                float factor = 1.0f / (2.0f * PI * puffs[p].sigma_x * puffs[p].sigma_y);
                float exponent = -0.5f * (
                    (dx * dx) / (puffs[p].sigma_x * puffs[p].sigma_x) +
                    (dy * dy) / (puffs[p].sigma_y * puffs[p].sigma_y)
                );

                if (exponent > -10.0f) {  // Cutoff for numerical stability
                    concentration += puffs[p].mass * factor * expf(exponent);
                }
            }
        }

        grid[iy * gridX + ix] = concentration;
    }
}

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPUFF-RCAPv3 - Working Version" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Check CUDA device
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA-capable GPU found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl << std::endl;

    // Initialize simulation parameters
    const int numPuffs = 100;
    const float dt = 1.0f;  // 1 second time step
    const int numSteps = 100;

    // Initialize grid for concentration
    const int gridX = 100;
    const int gridY = 100;
    const float cellSize = 10.0f;  // 10 meters per cell
    const float originX = -500.0f;
    const float originY = -500.0f;

    // Allocate host memory
    std::vector<PuffData> h_puffs(numPuffs);
    std::vector<float> h_grid(gridX * gridY, 0.0f);

    // Initialize puffs (release from origin)
    std::cout << "Initializing " << numPuffs << " puffs..." << std::endl;
    for (int i = 0; i < numPuffs; i++) {
        h_puffs[i].x = 0.0f;
        h_puffs[i].y = 0.0f;
        h_puffs[i].z = 10.0f + i * 2.0f;  // Stack vertically

        // Random wind direction
        float angle = (float)(rand() % 360) * PI / 180.0f;
        float speed = 5.0f + (rand() % 10);  // 5-15 m/s
        h_puffs[i].vx = speed * cosf(angle);
        h_puffs[i].vy = speed * sinf(angle);
        h_puffs[i].vz = 0.0f;

        // Initial dispersion
        h_puffs[i].sigma_x = 10.0f;
        h_puffs[i].sigma_y = 10.0f;
        h_puffs[i].sigma_z = 5.0f;

        h_puffs[i].mass = 1.0f;
        h_puffs[i].time = 0.0f;
        h_puffs[i].active = (i < numPuffs / 2);  // Only half are initially active
    }

    // Allocate device memory
    PuffData* d_puffs;
    float* d_grid;

    cudaMalloc(&d_puffs, numPuffs * sizeof(PuffData));
    cudaMalloc(&d_grid, gridX * gridY * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_puffs, h_puffs.data(), numPuffs * sizeof(PuffData), cudaMemcpyHostToDevice);

    // Run simulation
    std::cout << "\nRunning simulation for " << numSteps << " time steps..." << std::endl;

    for (int step = 0; step < numSteps; step++) {
        // Transport puffs
        int blockSize = 256;
        int numBlocks = (numPuffs + blockSize - 1) / blockSize;
        transportPuffs<<<numBlocks, blockSize>>>(d_puffs, numPuffs, dt);

        // Calculate concentration every 10 steps
        if (step % 10 == 0) {
            dim3 blockDim(16, 16);
            dim3 gridDim((gridX + blockDim.x - 1) / blockDim.x,
                        (gridY + blockDim.y - 1) / blockDim.y);

            calculateConcentration<<<gridDim, blockDim>>>(
                d_puffs, numPuffs, d_grid, gridX, gridY,
                cellSize, originX, originY
            );

            // Copy grid back and find maximum concentration
            cudaMemcpy(h_grid.data(), d_grid, gridX * gridY * sizeof(float), cudaMemcpyDeviceToHost);

            float maxConc = 0.0f;
            int maxX = 0, maxY = 0;
            for (int j = 0; j < gridY; j++) {
                for (int i = 0; i < gridX; i++) {
                    if (h_grid[j * gridX + i] > maxConc) {
                        maxConc = h_grid[j * gridX + i];
                        maxX = i;
                        maxY = j;
                    }
                }
            }

            std::cout << "Step " << step << ": Max concentration = " << maxConc
                     << " at grid(" << maxX << "," << maxY << ")" << std::endl;
        }
    }

    // Copy final puff data back
    cudaMemcpy(h_puffs.data(), d_puffs, numPuffs * sizeof(PuffData), cudaMemcpyDeviceToHost);

    // Print summary
    std::cout << "\nSimulation complete!" << std::endl;
    std::cout << "Final puff positions (first 5):" << std::endl;
    for (int i = 0; i < 5 && i < numPuffs; i++) {
        if (h_puffs[i].active) {
            std::cout << "  Puff " << i << ": ("
                     << h_puffs[i].x << ", "
                     << h_puffs[i].y << ", "
                     << h_puffs[i].z << ")" << std::endl;
        }
    }

    // Cleanup
    cudaFree(d_puffs);
    cudaFree(d_grid);

    std::cout << "\n========================================" << std::endl;
    std::cout << "GPUFF-RCAPv3 completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}