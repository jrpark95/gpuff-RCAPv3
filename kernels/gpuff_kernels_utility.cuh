// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel - Utility Functions
// ====================================================================================
//
// File: gpuff_kernels_utility.cuh
// Purpose: GPU kernels for debugging, printing, and auxiliary computations
//
// This file contains:
//   - Debug printing functions
//   - Simple dose computation variants
//   - Direct calculation methods
//   - Testing and validation kernels
//
// ====================================================================================

#ifndef GPUFF_KERNELS_UTILITY_CUH
#define GPUFF_KERNELS_UTILITY_CUH

#include "gpuff_kernels_constants.cuh"

// ====================================================================================
// CUDA Kernels - Debug Printing
// ====================================================================================

/**
 * Print simulation control data from device memory
 * Used for debugging and verification
 *
 * @param d_simControls Array of simulation control structures
 */
__global__ void use_data_in_device(SimulationControl* d_simControls) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < INPUT_NUM) {
        printf(" [Device Access]\n\nSimulation\t: %s\n\nRadial Distance\n\n",
            d_simControls[idx].sim_title);

        for (int i = 0; i < d_simControls[idx].numRad; ++i) {
            printf("%d)\t%.2f km\n", i+1, d_simControls[idx].ir_distances[i]);
        }
        printf("\n");
    }
}

/**
 * Print 2D array from device memory
 * Useful for debugging direction arrays
 *
 * @param d_dir Device array pointer
 * @param rows Number of rows
 * @param cols Number of columns
 */
__global__ void printDeviceArray(int* d_dir, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        printf("Row %d: ", row + 1);
        for (int col = 0; col < cols; ++col) {
            printf("%d ", d_dir[row * cols + col]);
        }
        printf("\n");
    }
}

/**
 * Print evacuee data from device memory
 * Used for debugging evacuee positions and states
 *
 * @param d_evacuees Array of evacuee structures
 * @param numEvacuees Total number of evacuees
 */
__global__ void printEvacueesKernel(const Evacuee* d_evacuees, size_t numEvacuees) {
    for (size_t idx = 0; idx < numEvacuees; ++idx) {
        printf("Evacuee %lu - Population: %f, Radius: %f, Theta: %f, Speed: %f\n",
            idx, d_evacuees[idx].population, d_evacuees[idx].r,
            d_evacuees[idx].theta, d_evacuees[idx].speed);
    }
}

// ====================================================================================
// CUDA Kernels - Simple Dose Computations
// ====================================================================================

/**
 * Compute evacuee dose using simple for loop
 * Basic implementation without optimizations
 *
 * @param d_puffs_RCAP Array of puff data
 * @param d_evacuees Array of evacuee data
 */
__global__ void computeEvacueeDoseForLoop(
    Gpuff::Puffcenter_RCAP * d_puffs_RCAP,
    Evacuee* d_evacuees
) {
    int simIdx = blockIdx.x;
    int evacueeIdx = threadIdx.x;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim) {
        float doseSum = 0.0f;
        for (int puffIdx = 0; puffIdx < d_totalpuff_per_Sim; ++puffIdx) {
            Gpuff::Puffcenter_RCAP& puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
            Evacuee& evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

            float dx = evacuee.r * cos(evacuee.theta) - puff.x;
            float dy = evacuee.r * sin(evacuee.theta) - puff.y;
            float distance = sqrt(dx * dx + dy * dy);

            if (distance > 0.0f) {
                float puffEffect = puff.conc[1] / (distance * distance);
                doseSum += puffEffect;
            }
        }
        d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose = doseSum * d_dt;
    }
}

/**
 * Compute evacuee dose using atomic operations
 * Alternative implementation with atomic accumulation
 *
 * @param d_puffs_RCAP Array of puff data
 * @param d_evacuees Array of evacuee data
 */
__global__ void computeEvacueeDoseAtomic(
    Gpuff::Puffcenter_RCAP* d_puffs_RCAP,
    Evacuee* d_evacuees
) {
    int simIdx = blockIdx.x;
    int evacueeIdx = threadIdx.x;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim) {
        for (int puffIdx = 0; puffIdx < d_totalpuff_per_Sim; ++puffIdx) {
            Gpuff::Puffcenter_RCAP& puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
            Evacuee& evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

            evacuee.dose = 0.0;

            float dx = evacuee.r * cos(evacuee.theta) - puff.x;
            float dy = evacuee.r * sin(evacuee.theta) - puff.y;
            float distance = sqrt(dx * dx + dy * dy);

            if (distance > 0.0f) {
                float puffEffect = puff.conc[1] / (distance * distance);

                atomicAdd(&(d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose), puffEffect * d_dt);
            }
        }
    }
}

// ====================================================================================
// CUDA Kernels - Reduction-Based Dose Computations
// ====================================================================================

/**
 * Compute dose for specific nuclides using reduction
 * Helper kernel for multi-level reduction approach
 *
 * @param puffs Puff concentration array
 * @param puffIdx Puff index
 * @param evacuees Evacuee data
 * @param evacueeIdx Evacuee index
 * @param exposure Exposure coefficients
 * @param distance Distance between puff and evacuee
 * @param sdata Shared memory for reduction
 * @param nuclideStart Starting nuclide index
 * @param nuclideEnd Ending nuclide index
 */
__global__ void computeDoseForNuclide(
    float* puffs, int puffIdx, Evacuee evacuees, int evacueeIdx, float* exposure,
    float distance, float* sdata, int nuclideStart, int nuclideEnd) {
    extern __shared__ float localDose[];
    localDose[threadIdx.x] = 0.0f;

    for (int nuclideIdx = nuclideStart + threadIdx.x; nuclideIdx < nuclideEnd; nuclideIdx += blockDim.x) {
        float puffEffect = puffs[nuclideIdx] / (distance * distance);
        float totalExposure = 0.0;
        for (int organIdx = 0; organIdx < MAX_ORGANS; ++organIdx) {
            totalExposure += exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 1];
        }
        if (puffEffect * totalExposure > 0.0f) {
            localDose[threadIdx.x] += puffEffect * totalExposure;
        }
    }

    __syncthreads();

    // Reduce within this block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            localDose[threadIdx.x] += localDose[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&sdata[0], localDose[0] * d_dt);
    }
}

/**
 * Compute evacuee dose using reduction pattern
 * Optimized version with shared memory reduction
 *
 * @param d_puffs_RCAP Array of puff data
 * @param d_evacuees Array of evacuee data
 * @param d_exposure Exposure coefficients
 */
__global__ void computeEvacueeDoseReduction1(
    Gpuff::Puffcenter_RCAP* d_puffs_RCAP,
    Evacuee* d_evacuees, float* d_exposure
) {
    extern __shared__ float sdata[];

    int simIdx = blockIdx.x;
    int evacueeIdx = blockIdx.y;
    int puffIdx = threadIdx.x;

    if (simIdx < d_numSims&& evacueeIdx < d_totalevacuees_per_Sim) {
        sdata[threadIdx.x] = 0.0f;

        Gpuff::Puffcenter_RCAP& puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee& evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

        float dx = evacuee.r * cos(evacuee.theta) - puff.x;
        float dy = evacuee.r * sin(evacuee.theta) - puff.y;
        float distance = sqrt(dx * dx + dy * dy);

        if (distance > 0.0f) {
            float doseSum = 0.0f;
            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffEffect = puff.conc[nuclideIdx] / (distance * distance);
                float exposure = 0.0;

                for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++)
                    exposure += d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 1];

                if (puffEffect * exposure > 0.0f) doseSum += puffEffect * exposure;
            }
            sdata[threadIdx.x] = doseSum * d_dt;
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose = sdata[0];
        }
    }
}

// ====================================================================================
// CUDA Kernels - Direct Inhalation
// ====================================================================================

/**
 * Direct inhalation dose calculation
 * Simplified calculation focusing only on inhalation pathway
 *
 * @param d_puffs_RCAP Array of puff data
 * @param d_evacuees Array of evacuee data
 * @param d_exposure Exposure coefficients
 * @param dPF Protection factors
 */
__global__ void DirectInhalation(
    Gpuff::Puffcenter_RCAP* d_puffs_RCAP,
    Evacuee* d_evacuees, float* d_exposure, const ProtectionFactors* dPF
) {
    extern __shared__ float sdata[];

    int simIdx = blockIdx.x;
    int evacueeIdx = blockIdx.y;
    int puffIdx = threadIdx.x;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim) {
        sdata[threadIdx.x] = 0.0f;

        Gpuff::Puffcenter_RCAP& puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee& evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

        float dx = evacuee.r * cos(evacuee.theta) - puff.x;
        float dy = evacuee.r * sin(evacuee.theta) - puff.y;
        float distance = sqrt(dx * dx + dy * dy);

        if (distance > 0.0f) {
            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffEffect = puff.conc[nuclideIdx] / (distance * distance);

                if (puffEffect > 0.0f) {
                    float totalExposure = 0.0;
                    for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++) {
                        if (d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2] > 0.0f) {
                            totalExposure += d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2];
                        }
                    }

                    sdata[threadIdx.x] += puffEffect * totalExposure * d_dt;
                }
            }
        }

        sdata[threadIdx.x] *= dPF->pfactor[puff.flag][4]* dPF->pfactor[puff.flag][2];

        __syncthreads();

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sdata[threadIdx.x] += __shfl_down_sync(0xffffffff, sdata[threadIdx.x], offset);
        }

        if (threadIdx.x == 0) {
            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose = sdata[0];
        }
    }
}

// ====================================================================================
// CUDA Kernels - Decay Ground Deposit
// ====================================================================================

/**
 * Apply radioactive decay to ground deposition
 * Updates ground deposit values based on decay constants
 *
 * @param ground_deposit Ground deposition array
 * @param d_ND Nuclide data containing half-lives
 * @param numTheta Number of angular sectors
 * @param numRad Number of radial rings
 */
__global__ void decayGroundDeposit(float* ground_deposit, NuclideData* d_ND, int numTheta, int numRad) {
    int theta_idx = blockIdx.x;
    int rad_idx = threadIdx.x;

    if (theta_idx >= numTheta || rad_idx >= numRad) return;

    for (int nuc_idx = 0; nuc_idx < MAX_NUCLIDES; ++nuc_idx) {
        int deposit_idx = theta_idx * numRad * MAX_NUCLIDES +
                         rad_idx * MAX_NUCLIDES + nuc_idx;

        float current_deposit = ground_deposit[deposit_idx];
        if (current_deposit > 0.0f) {
            float half_life = d_ND[nuc_idx].half_life;
            if (half_life > 0.0f) {
                float decay_factor = expf(-logf(2.0f) / half_life * d_dt);
                ground_deposit[deposit_idx] *= decay_factor;
            }
        }
    }
}

#endif // GPUFF_KERNELS_UTILITY_CUH