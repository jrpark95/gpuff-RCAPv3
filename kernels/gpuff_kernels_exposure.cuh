// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel - Exposure Calculations
// ====================================================================================
//
// File: gpuff_kernels_exposure.cuh
// Purpose: GPU kernels for radiation exposure calculations
//
// This file contains:
//   - Inhalation dose calculations
//   - Cloudshine (external) dose calculations
//   - Organ-specific dose accumulation
//   - Protection factor applications
//
// ====================================================================================

#ifndef GPUFF_KERNELS_EXPOSURE_CUH
#define GPUFF_KERNELS_EXPOSURE_CUH

#include "gpuff_kernels_constants.cuh"

// ====================================================================================
// CUDA Kernels - Basic Exposure
// ====================================================================================

/**
 * Compute radiation exposure for evacuees from puff cloud
 * Calculates both inhalation and cloudshine doses
 *
 * Thread organization:
 *   - blockIdx.x: Simulation index
 *   - blockIdx.y: Evacuee index
 *   - threadIdx.x: Puff index
 *
 * @param d_puffs_RCAP Array of puff data
 * @param d_evacuees Array of evacuee data
 * @param d_exposure Exposure coefficients by nuclide/organ
 * @param dPF Protection factors for different states
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
        //float gaussianFactor = __expf(exponent) / (2.0f * PI * sigma_h * sigma_z);
        float gaussianFactor = 1.0 / (dx * dx + dy * dy);


        if (1) {
        //if (gaussianFactor > 1e-10f) {

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

// ====================================================================================
// CUDA Kernels - Organ-Specific Exposure
// ====================================================================================

/**
 * Compute organ-specific radiation exposure with mixing height correction
 * Calculates doses for each organ separately
 *
 * Uses:
 *   - Gaussian dispersion for inhalation
 *   - Point source approximation for cloudshine
 *   - Protection factors based on evacuee state
 *
 * @param d_puffs_RCAP Array of puff data
 * @param d_evacuees Array of evacuee data
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

        // Accumulate doses for each organ
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
                d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_inhalations[organIdx] += sdata_inhalation[threadIdx.x];
                d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_cloudshines[organIdx] += sdata_cloudshine[threadIdx.x];
            }
            __syncthreads();

        }

        if (sdata_inhalation[threadIdx.x] > 0.000)
            printf("%d, %d\n", simIdx, evacueeIdx);

    }
}

// ====================================================================================
// CUDA Kernels - XY Coordinate Exposure
// ====================================================================================

/**
 * Compute exposure using XY Cartesian coordinates
 * Simplified version focusing on inhalation dose only
 *
 * @param d_puffs_RCAP Array of puff data
 * @param d_evacuees Array of evacuee data
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
 * Single-unit exposure calculation in XY coordinates
 * Processes only unit 0 puffs
 *
 * @param d_puffs_RCAP Array of puff data
 * @param d_evacuees Array of evacuee data
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

#endif // GPUFF_KERNELS_EXPOSURE_CUH