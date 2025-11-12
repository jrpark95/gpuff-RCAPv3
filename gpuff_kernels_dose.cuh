// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel Implementation - Dose and Exposure Calculations
// ====================================================================================
//
// File: gpuff_kernels_dose.cuh
// Purpose: Evacuee dose assessment and exposure calculation kernels
//
// This file contains CUDA kernels for:
//   - Direct inhalation dose calculations
//   - Cloudshine external exposure
//   - Organ-specific dose computations
//   - Protection factor applications
//   - Ground deposition decay
//   - Debug and utility functions
//
// ====================================================================================

#ifndef GPUFF_KERNELS_DOSE_CUH
#define GPUFF_KERNELS_DOSE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "gpuff_struct.cuh"

// Note: All device constants and variables (d_nop, d_dt, d_numSims, d_totalevacuees_per_Sim, d_totalpuff_per_Sim)
// are defined in gpuff.cuh which is included through the include chain
// No extern declarations needed here

// ====================================================================================
// Utility and Debug Kernels
// ====================================================================================

/**
 * Access and display simulation control data on device
 * Used for debugging and verification
 *
 * @param d_simControls Simulation control structures
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
 * Print device array for debugging
 *
 * @param d_dir Direction array
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
 * Print evacuee data for debugging
 *
 * @param d_evacuees Evacuee array
 * @param numEvacuees Number of evacuees
 */
__global__ void printEvacueesKernel(const Evacuee* d_evacuees, size_t numEvacuees) {
    for (size_t idx = 0; idx < numEvacuees; ++idx) {
        printf("Evacuee %lu - Population: %f, Radius: %f, Theta: %f, Speed: %f\n",
            idx, d_evacuees[idx].population, d_evacuees[idx].r,
            d_evacuees[idx].theta, d_evacuees[idx].speed);
    }
}

// ====================================================================================
// Basic Dose Computation Kernels
// ====================================================================================

/**
 * Compute evacuee dose using for loops
 * Simple implementation for validation
 *
 * @param d_puffs_RCAP Puff centers
 * @param d_evacuees Evacuee array
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
 *
 * @param d_puffs_RCAP Puff centers
 * @param d_evacuees Evacuee array
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
// Advanced Dose Computation with Reduction
// ====================================================================================

/**
 * Compute dose for specific nuclide using shared memory reduction
 *
 * @param puffs Puff concentrations
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
 * Compute evacuee dose with shared memory reduction
 *
 * @param d_puffs_RCAP Puff centers
 * @param d_evacuees Evacuee array
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
// Inhalation Dose Calculations
// ====================================================================================

/**
 * Direct inhalation dose calculation with protection factors
 *
 * @param d_puffs_RCAP Puff centers
 * @param d_evacuees Evacuee array
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

        // Warp shuffle reduction
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sdata[threadIdx.x] += __shfl_down_sync(0xffffffff, sdata[threadIdx.x], offset);
        }

        if (threadIdx.x == 0) {
            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose = sdata[0];
        }
    }
}

/**
 * Optimized direct inhalation calculation with fast math
 *
 * @param d_puffs_RCAP Puff centers
 * @param d_evacuees Evacuee array
 * @param d_exposure Exposure coefficients
 * @param dPF Protection factors
 */
__global__ void DirectInhalation1(
    Gpuff::Puffcenter_RCAP* __restrict__ d_puffs_RCAP,
    Evacuee* __restrict__ d_evacuees,
    const float* __restrict__ d_exposure,
    const ProtectionFactors* __restrict__ dPF
) {
    extern __shared__ float sdata[];

    int simIdx = blockIdx.x;
    int evacueeIdx = blockIdx.y;
    int puffIdx = threadIdx.x;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim) {
        sdata[threadIdx.x] = 0.0f;

        Gpuff::Puffcenter_RCAP puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

        float cosTheta = __cosf(evacuee.theta);
        float sinTheta = __sinf(evacuee.theta);

        float dx = evacuee.r * cosTheta - puff.x;
        float dy = evacuee.r * sinTheta - puff.y;
        float distanceSq = dx * dx + dy * dy;

        if (distanceSq > 0.0f) {
            float invDistanceSq = __frcp_rn(distanceSq);
            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffEffect = puff.conc[nuclideIdx] * invDistanceSq;

                if (puffEffect > 0.0f) {
                    float totalExposure = 0.0f;

#pragma unroll
                    for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++) {
                        float exposureValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2];
                        if (exposureValue > 0.0f) {
                            totalExposure += exposureValue;
                        }
                    }

                    sdata[threadIdx.x] += puffEffect * totalExposure * d_dt;
                }
            }
        }

        sdata[threadIdx.x] *= dPF->pfactor[puff.flag][4] * dPF->pfactor[puff.flag][2];

        __syncthreads();

        float val = sdata[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (threadIdx.x == 0) {
            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose = val;
        }
    }
}

// ====================================================================================
// Combined Exposure Calculations (Inhalation + Cloudshine)
// ====================================================================================

/**
 * Compute both inhalation and cloudshine exposure
 *
 * @param d_puffs_RCAP Puff centers
 * @param d_evacuees Evacuee array
 * @param d_exposure Exposure coefficients
 * @param dPF Protection factors
 */
__global__ void ComputeExposure(
    Gpuff::Puffcenter_RCAP* __restrict__ d_puffs_RCAP,
    Evacuee* __restrict__ d_evacuees,
    const float* __restrict__ d_exposure,
    const ProtectionFactors* __restrict__ dPF
) {
    extern __shared__ float sdata[];

    int simIdx = blockIdx.x;
    int evacueeIdx = blockIdx.y;
    int puffIdx = threadIdx.x;

    float* sdata_inhalation = sdata;
    float* sdata_cloudshine = sdata + blockDim.x;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim && puffIdx < d_totalpuff_per_Sim) {
        sdata_inhalation[threadIdx.x] = 0.0f;
        sdata_cloudshine[threadIdx.x] = 0.0f;

        Gpuff::Puffcenter_RCAP puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

        float cosTheta = __cosf(evacuee.theta);
        float sinTheta = __sinf(evacuee.theta);

        float dx = evacuee.r * cosTheta - puff.x;
        float dy = evacuee.r * sinTheta - puff.y;
        float dz = puff.z;

        float sigma_h = puff.sigma_h;
        float sigma_z = puff.sigma_z;

        float exponent = -(dx * dx + dy * dy) / (2.0f * sigma_h * sigma_h)
            - (dz * dz) / (2.0f * sigma_z * sigma_z);
        float gaussianFactor = 1.0 / (dx * dx + dy * dy);

        if (1) {
            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffConc = puff.conc[nuclideIdx] * gaussianFactor;

                if (puffConc > 0.0f) {
                    float totalInhalation = 0.0f;
                    float totalCloudshine = 0.0f;

#pragma unroll
                    for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++) {

                        float cloudshineValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 0];
                        if (cloudshineValue > 0.0f) {
                            totalCloudshine += cloudshineValue;
                        }

                        float inhalationValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2];
                        if (inhalationValue > 0.0f) {
                            totalInhalation += inhalationValue;
                        }
                    }

                    sdata_inhalation[threadIdx.x] += puffConc * totalInhalation * d_dt;
                    sdata_cloudshine[threadIdx.x] += puffConc * totalCloudshine * d_dt;
                }
            }

            sdata_inhalation[threadIdx.x] *= dPF->pfactor[puff.flag][4];
            sdata_cloudshine[threadIdx.x] *= dPF->pfactor[puff.flag][2];
        }

        __syncthreads();

        float inhalationDose = sdata_inhalation[threadIdx.x];
        float cloudshineDose = sdata_cloudshine[threadIdx.x];

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            inhalationDose += __shfl_down_sync(0xffffffff, inhalationDose, offset);
            cloudshineDose += __shfl_down_sync(0xffffffff, cloudshineDose, offset);
        }

        if (threadIdx.x % warpSize == 0) {
            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_inhalation = inhalationDose;
            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_cloudshine = cloudshineDose;
        }
    }
}

/**
 * Compute exposure with mixing height consideration and organ-specific doses
 *
 * @param d_puffs_RCAP Puff centers
 * @param d_evacuees Evacuee array
 * @param d_exposure Exposure coefficients
 * @param dPF Protection factors
 */
__global__ void ComputeExposureHmix(
    Gpuff::Puffcenter_RCAP* __restrict__ d_puffs_RCAP,
    Evacuee* __restrict__ d_evacuees,
    const float* __restrict__ d_exposure,
    const ProtectionFactors* __restrict__ dPF
) {
    extern __shared__ float sdata[];

    int simIdx = blockIdx.y;
    int evacueeIdx = blockIdx.x;
    int puffIdx = threadIdx.x;

    float hmix = 1500.0;

    float* sdata_inhalation = sdata;
    float* sdata_cloudshine = sdata + blockDim.x;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim && puffIdx < d_totalpuff_per_Sim) {
        sdata_inhalation[threadIdx.x] = 0.0f;
        sdata_cloudshine[threadIdx.x] = 0.0f;

        Gpuff::Puffcenter_RCAP puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

        float cosTheta = cosf(evacuee.theta);
        float sinTheta = sinf(evacuee.theta);

        float inhalationDose = 0.0f;
        float cloudshineDose = 0.0f;

        float dx = evacuee.r * cosTheta - puff.x;
        float dy = evacuee.r * sinTheta - puff.y;
        float z_evac = 0.0;
        float H = puff.z;

        float sigma_h = puff.sigma_h;
        float sigma_z = puff.sigma_z;

        float gaussianFactor = (1.0f / (powf(2.0f * PI, 1.5f) * sigma_h * sigma_h * sigma_z)) *
            expf(-(dx * dx) / (2.0f * sigma_h * sigma_h)
                - (dy * dy) / (2.0f * sigma_h * sigma_h));

        float distanceFactor = 1 / (4.0f * PI * (dx * dx + dy * dy + H * H));

        int pfidx = 0;
        if (evacuee.flag == 0) pfidx = 1;
        else if (evacuee.flag == 1) pfidx = 2;
        else if (evacuee.flag == 2) pfidx = 0;

        float puffInhalationDose[MAX_ORGANS] = { 0.0f, };
        float puffCloudshineDose[MAX_ORGANS] = { 0.0f, };

        if (1) {
            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffConc = puff.conc[nuclideIdx];

                if (puffConc > 0.0f) {
                    float totalInhalation[MAX_ORGANS] = { 0.0f, };
                    float totalCloudshine[MAX_ORGANS] = { 0.0f, };

#pragma unroll
                    for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++) {

                        float cloudshineValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 0];
                        if (cloudshineValue > 0.0f) {
                            puffCloudshineDose[organIdx] += cloudshineValue * puffConc * d_dt * distanceFactor;
                            if (dPF->pfactor[pfidx][0] > 0) puffCloudshineDose[organIdx] *= dPF->pfactor[pfidx][0];
                        }

                        float inhalationValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2];
                        if (inhalationValue > 0.0f) {
                            puffInhalationDose[organIdx] += inhalationValue * puffConc * d_dt * gaussianFactor;
                            if (dPF->pfactor[pfidx][4] > 0) puffInhalationDose[organIdx] *= dPF->pfactor[pfidx][4] * dPF->pfactor[pfidx][2];
                        }
                    }
                }
            }
        }

        // Track total cloudshine for new fields
        float total_cloudshine_instant = 0.0f;

        for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++) {

            if (puffInhalationDose[organIdx] > 0) sdata_inhalation[threadIdx.x] = puffInhalationDose[organIdx];
            else sdata_inhalation[threadIdx.x] = 0;

            if (puffCloudshineDose[organIdx] > 0) sdata_cloudshine[threadIdx.x] = puffCloudshineDose[organIdx];
            else sdata_cloudshine[threadIdx.x] = 0;

            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    sdata_inhalation[threadIdx.x] += sdata_inhalation[threadIdx.x + s];
                    sdata_cloudshine[threadIdx.x] += sdata_cloudshine[threadIdx.x + s];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                int evac_idx = simIdx * d_totalevacuees_per_Sim + evacueeIdx;
                d_evacuees[evac_idx].dose_inhalations[organIdx] += sdata_inhalation[threadIdx.x];
                d_evacuees[evac_idx].dose_cloudshines[organIdx] += sdata_cloudshine[threadIdx.x];

                // Sum up cloudshine for all organs
                if (sdata_cloudshine[threadIdx.x] > 0.0f) {
                    total_cloudshine_instant += sdata_cloudshine[threadIdx.x];
                }
            }
            __syncthreads();
        }

        // Update new cloudshine tracking fields after processing all organs
        if (threadIdx.x == 0 && total_cloudshine_instant > 0.0f) {
            int evac_idx = simIdx * d_totalevacuees_per_Sim + evacueeIdx;
            d_evacuees[evac_idx].dose_cloudshine_cumulative += total_cloudshine_instant;
            d_evacuees[evac_idx].dose_cloudshine_instant = total_cloudshine_instant;
            // Determine cloudshine mode based on puff size
            if (sigma_h < 400.0f && sigma_z < 400.0f) {
                d_evacuees[evac_idx].cloudshine_mode = 0; // small_puff
            } else if (sigma_z < 400.0f) {
                d_evacuees[evac_idx].cloudshine_mode = 1; // plane_source
            } else {
                d_evacuees[evac_idx].cloudshine_mode = 2; // semi_infinite
            }
        }

        if (sdata_inhalation[threadIdx.x] > 0.000)
            printf("%d, %d\n", simIdx, evacueeIdx);
    }
}

// ====================================================================================
// Specialized Exposure Calculations
// ====================================================================================

/**
 * Compute exposure for XY coordinate system
 *
 * @param d_puffs_RCAP Puff centers
 * @param d_evacuees Evacuee array
 * @param d_exposure Exposure coefficients
 * @param dPF Protection factors
 */
__global__ void ComputeExposureHmix_xy(
    Gpuff::Puffcenter_RCAP* __restrict__ d_puffs_RCAP,
    Evacuee* __restrict__ d_evacuees,
    const float* __restrict__ d_exposure,
    const ProtectionFactors* __restrict__ dPF
) {
    extern __shared__ float sdata[];

    int simIdx = blockIdx.y;
    int evacueeIdx = blockIdx.x;
    int puffIdx = threadIdx.x;

    float hmix = 1500.0;

    float* sdata_inhalation = sdata;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim && puffIdx < d_totalpuff_per_Sim) {
        sdata_inhalation[threadIdx.x] = 0.0f;

        Gpuff::Puffcenter_RCAP puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

        float inhalationDose = 1.0e-40;

        float dx = evacuee.x - puff.x;
        float dy = evacuee.y - puff.y;
        float z_evac = 0.0;
        float H = puff.z;

        float sigma_h = puff.sigma_h;
        float sigma_z = puff.sigma_z;

        float gaussianFactor = (1.0f / (powf(2.0f * PI, 1.5f) * sigma_h * sigma_h * sigma_z)) *
            expf(-(dx * dx) / (2.0f * sigma_h * sigma_h)
                - (dy * dy) / (2.0f * sigma_h * sigma_h));

        float distanceFactor = 1 / (4.0f * PI * (dx * dx + dy * dy + H * H));

        if (gaussianFactor > 1e-26f) {
            float puffInhalationDose = 1.0e-40;
            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffConc = puff.conc[nuclideIdx];

                if (puffConc > 0.0f) {
                    float totalInhalation = 1.0e-40;

#pragma unroll
                    for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++) {
                        float inhalationValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2];
                        if (inhalationValue > 0.0f) {
                            totalInhalation += inhalationValue;
                        }
                    }

                    puffInhalationDose += puffConc * totalInhalation * d_dt;
                }
            }

            if (dPF->pfactor[puff.flag][4] > 0) puffInhalationDose *= dPF->pfactor[puff.flag][4];

            inhalationDose += puffInhalationDose * gaussianFactor;
        }

        if (inhalationDose > 0) sdata_inhalation[threadIdx.x] = inhalationDose;
        else sdata_inhalation[threadIdx.x] = 1.0e-40;

        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata_inhalation[threadIdx.x] += sdata_inhalation[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_inhalation = sdata_inhalation[threadIdx.x];
        }
    }
}

/**
 * Compute exposure for single unit
 *
 * @param d_puffs_RCAP Puff centers
 * @param d_evacuees Evacuee array
 * @param d_exposure Exposure coefficients
 * @param dPF Protection factors
 */
__global__ void ComputeExposureHmix_xy_single(
    Gpuff::Puffcenter_RCAP* __restrict__ d_puffs_RCAP,
    Evacuee* __restrict__ d_evacuees,
    const float* __restrict__ d_exposure,
    const ProtectionFactors* __restrict__ dPF
) {
    extern __shared__ float sdata[];

    int simIdx = blockIdx.y;
    int evacueeIdx = blockIdx.x;
    int puffIdx = threadIdx.x;

    float hmix = 1500.0;

    float* sdata_inhalation = sdata;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim && puffIdx < d_totalpuff_per_Sim) {
        sdata_inhalation[threadIdx.x] = 0.0f;

        Gpuff::Puffcenter_RCAP puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

        if (puff.unitidx != 0) return;

        float inhalationDose = 1.0e-40;

        float dx = evacuee.x - puff.x;
        float dy = evacuee.y - puff.y;
        float z_evac = 0.0;
        float H = puff.z;

        float sigma_h = puff.sigma_h;
        float sigma_z = puff.sigma_z;

        float gaussianFactor = (1.0f / (powf(2.0f * PI, 1.5f) * sigma_h * sigma_h * sigma_z)) *
            expf(-(dx * dx) / (2.0f * sigma_h * sigma_h)
                - (dy * dy) / (2.0f * sigma_h * sigma_h));

        float distanceFactor = 1 / (4.0f * PI * (dx * dx + dy * dy + H * H));

        if (gaussianFactor > 1e-26f) {
            float puffInhalationDose = 1.0e-40;
            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffConc = puff.conc[nuclideIdx];

                if (puffConc > 0.0f) {
                    float totalInhalation = 1.0e-40;

#pragma unroll
                    for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++) {
                        float inhalationValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2];
                        if (inhalationValue > 0.0f) {
                            totalInhalation += inhalationValue;
                        }
                    }

                    puffInhalationDose += puffConc * totalInhalation * d_dt;
                }
            }

            if (dPF->pfactor[puff.flag][4] > 0) puffInhalationDose *= dPF->pfactor[puff.flag][4];

            inhalationDose += puffInhalationDose * gaussianFactor;
        }

        if (inhalationDose > 0) sdata_inhalation[threadIdx.x] = inhalationDose;
        else sdata_inhalation[threadIdx.x] = 1.0e-40;

        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata_inhalation[threadIdx.x] += sdata_inhalation[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_cloudshine = 6.0 * sdata_inhalation[threadIdx.x];
        }
    }
}

// ====================================================================================
// Additional Dose Computation Kernels
// ====================================================================================

/**
 * Compute evacuee dose reduction with XY indexing
 *
 * @param d_puffs_RCAP Puff centers
 * @param d_evacuees Evacuee array
 * @param d_exposure Exposure coefficients
 */
__global__ void computeEvacueeDoseReductionXY(
    Gpuff::Puffcenter_RCAP* d_puffs_RCAP,
    Evacuee* d_evacuees, float* d_exposure
) {
    extern __shared__ float sdata[];

    int simIdx = blockIdx.x;
    int evacueeIdx = blockIdx.y;
    int puffIdx = threadIdx.x;
    int organIdx = threadIdx.y;

    // Calculate the index for shared memory
    int sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim) {
        sdata[sharedIndex] = 0.0f;

        Gpuff::Puffcenter_RCAP& puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee& evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

        float dx = evacuee.r * cos(evacuee.theta) - puff.x;
        float dy = evacuee.r * sin(evacuee.theta) - puff.y;
        float distance = sqrt(dx * dx + dy * dy);

        if (distance > 0.0f) {
            float doseSum = 0.0f;

            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffEffect = puff.conc[nuclideIdx] / (distance * distance);
                float exposure = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 1];

                if (puffEffect * exposure > 0.0f) doseSum += puffEffect * exposure;
            }
            sdata[sharedIndex] = doseSum * d_dt;
        }

        __syncthreads();

        // Perform parallel reduction in shared memory
        for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (sharedIndex < s) {
                sdata[sharedIndex] += sdata[sharedIndex + s];
            }
            __syncthreads();
        }

        // Write result for this block to global mem
        if (sharedIndex == 0) {
            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose += sdata[0];
        }
    }
}

/**
 * Compute evacuee dose reduction with atomic operations
 *
 * @param d_puffs_RCAP Puff centers
 * @param d_evacuees Evacuee array
 * @param d_exposure Exposure coefficients
 */
__global__ void computeEvacueeDoseReductionAtomic(
    Gpuff::Puffcenter_RCAP* d_puffs_RCAP,
    Evacuee* d_evacuees, float* d_exposure
) {
    int simIdx = blockIdx.x;
    int evacueeIdx = blockIdx.y;
    int puffIdx = threadIdx.x;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim) {
        float doseSum = 0.0f;
        Gpuff::Puffcenter_RCAP& puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee& evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];
        float dx = evacuee.r * cos(evacuee.theta) - puff.x;
        float dy = evacuee.r * sin(evacuee.theta) - puff.y;
        float distance = sqrt(dx * dx + dy * dy);

        if (distance > 0.0f) {
            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffEffect = puff.conc[nuclideIdx] / (distance * distance);
                float exposure = 0.0f;
                for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++) {
                    exposure += d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 1];
                }
                doseSum += puffEffect * exposure;
            }
        }

        atomicAdd(&d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose, doseSum * d_dt);
    }
}

// ====================================================================================
// Nuclide Data Management
// ====================================================================================

/**
 * Print nuclide data for debugging
 *
 * @param d_ND Nuclide data array
 */
__global__ void printNuclideData(NuclideData* d_ND) {
    for (int i = 0; i < MAX_NUCLIDES; ++i) {
        NuclideData nuclide = d_ND[i];

        printf("Nuclide %d:\n", i);
        printf("  Name: %s\n", nuclide.name);
        printf("  ID: %d\n", nuclide.id);
        printf("  Half-life: %f\n", nuclide.half_life);
        printf("  Atomic weight: %f\n", nuclide.atomic_weight);
        printf("  Chemical group: %d\n", nuclide.chemical_group);
        printf("  Wet deposition: %f\n", nuclide.wet_deposition);
        printf("  Dry deposition: %f\n", nuclide.dry_deposition);
        printf("  Core inventory: %f\n", nuclide.core_inventory);
        printf("  Decay count: %d\n", nuclide.decay_count);
        printf("  Organ count: %d\n", nuclide.organ_count);

        printf("\n");
    }
}

/**
 * Apply radioactive decay to ground deposit
 *
 * @param ground_deposit Ground deposition array
 * @param d_ND Nuclide data
 * @param numTheta Number of angular sectors
 * @param numRad Number of radial zones
 */
__global__ void decayGroundDeposit(float* ground_deposit, NuclideData* d_ND, int numTheta, int numRad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = numTheta * numRad * MAX_NUCLIDES;

    if (idx < total_size) {
        int nuc_idx = idx % MAX_NUCLIDES;
        NuclideData nuclide = d_ND[nuc_idx];
        float decay_factor = expf(-logf(2.0f) / nuclide.half_life * d_dt);
        ground_deposit[idx] *= decay_factor;
    }
}

// ====================================================================================
// CPU Validation Functions
// ====================================================================================

/**
 * CPU version of ComputeExposureHmix for validation
 */
void Gpuff::ComputeExposureHmix_cpu(
    std::vector<Evacuee> evacuees,
    ProtectionFactors PF,
    int numSims,
    int totalEvacueesPerSim,
    int totalPuffsPerSim
) {
    float hmix = 1500.0;

    for (int simIdx = 0; simIdx < numSims; ++simIdx) {
        for (int evacueeIdx = 0; evacueeIdx < totalEvacueesPerSim; ++evacueeIdx) {

            float sum_inhal = 0.0f;
            float sum_cshine = 0.0f;

            for (int puffIdx = 0; puffIdx < totalPuffsPerSim; ++puffIdx) {
                Gpuff::Puffcenter_RCAP puff = puffs_RCAP[simIdx * totalPuffsPerSim + puffIdx];
                Evacuee evacuee = evacuees[simIdx * totalEvacueesPerSim + evacueeIdx];

                float inhal = 0.0f;
                float cshine = 0.0f;

                if (puff.flag == 0) {
                    continue;
                }

                float cosTheta = cos(evacuee.theta);
                float sinTheta = sin(evacuee.theta);

                float dx = evacuee.r * cosTheta - puff.x;
                float dy = evacuee.r * sinTheta - puff.y;
                float z_evac = 0.0;
                float H = puff.z;

                float sigma_h = puff.sigma_h;
                float sigma_z = puff.sigma_z;

                float gaussianFactor = (1.0f / (pow(2.0f * PI, 1.5f) * sigma_h * sigma_h * sigma_z)) *
                    expf(-(dx * dx + dy * dy) / (2.0f * sigma_h * sigma_h));

                float distanceFactor = 1 / (4.0f * PI * (dx * dx + dy * dy + H * H));

                float inhalationDose = 0.0f;
                float cloudshineDose = 0.0f;

                if (1) {
                    for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                        float puffConc = puff.conc[nuclideIdx];

                        if (puffConc > 0.0f) {
                            float totalInhalation = 0.0f;
                            float totalCloudshine = 0.0f;

                            for (int organIdx = 0; organIdx < MAX_ORGANS; ++organIdx) {
                                float cloudshineValue = exposure_data_all[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 0];
                                if (cloudshineValue > 0.0f) {
                                    totalCloudshine += cloudshineValue;
                                }

                                float inhalationValue = exposure_data_all[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2];
                                if (inhalationValue > 0.0f) {
                                    totalInhalation += inhalationValue;
                                }
                            }

                            inhalationDose += puffConc * totalInhalation * dt;
                            cloudshineDose += puffConc * totalCloudshine * dt;
                        }
                    }

                    inhalationDose *= PF.pfactor[puff.flag][4];
                    cloudshineDose *= PF.pfactor[puff.flag][2];

                    printf("flag=%d, factor=%f\n", puff.flag, PF.pfactor[puff.flag][4]);
                }

                sum_inhal += inhalationDose * gaussianFactor;
                sum_cshine += cloudshineDose * distanceFactor;
            }

            evacuees[simIdx * totalEvacueesPerSim + evacueeIdx].dose_inhalation = sum_inhal;
            evacuees[simIdx * totalEvacueesPerSim + evacueeIdx].dose_cloudshine = sum_cshine;

            if (simIdx == 0 && evacueeIdx == 4) {
                printf("%e\n", evacuees[simIdx * totalevacuees_per_Sim + evacueeIdx].dose_inhalation);
            }
        }
    }
}

#endif // GPUFF_KERNELS_DOSE_CUH