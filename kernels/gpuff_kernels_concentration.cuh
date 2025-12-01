// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel - Concentration Calculations
// ====================================================================================
//
// File: gpuff_kernels_concentration.cuh
// Purpose: GPU kernels for concentration accumulation and spatial extent calculations
//
// This file contains:
//   - Grid-based concentration accumulation
//   - Receptor point concentration calculations
//   - Spatial extent (min/max) finding
//   - Gaussian puff concentration formulas
//
// ====================================================================================

#ifndef GPUFF_KERNELS_CONCENTRATION_CUH
#define GPUFF_KERNELS_CONCENTRATION_CUH

#include "gpuff_kernels_constants.cuh"

// ====================================================================================
// CUDA Kernels - Spatial Extent
// ====================================================================================

/**
 * Find minimum and maximum X,Y coordinates of puff cloud
 * Uses parallel reduction with shared memory for efficiency
 *
 * Thread organization: 1D grid, uses shared memory reduction
 * Memory access: Coalesced reads, atomic operations for final reduction
 *
 * @param d_puffs Array of puff center data
 * @param d_minX Device pointer to store minimum X
 * @param d_minY Device pointer to store minimum Y
 * @param d_maxX Device pointer to store maximum X
 * @param d_maxY Device pointer to store maximum Y
 */
__global__ void findMinMax(
    Gpuff::Puffcenter* d_puffs,
    float* d_minX, float* d_minY,
    float* d_maxX, float* d_maxY)
{

    extern __shared__ float sharedData[];
    float* s_minX = sharedData;
    float* s_minY = &sharedData[blockDim.x];
    float* s_maxX = &sharedData[2 * blockDim.x];
    float* s_maxY = &sharedData[3 * blockDim.x];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    s_minX[tid] = (index < d_nop) ? d_puffs[index].x : FLT_MAX;
    s_minY[tid] = (index < d_nop) ? d_puffs[index].y : FLT_MAX;
    s_maxX[tid] = (index < d_nop) ? d_puffs[index].x : -FLT_MAX;
    s_maxY[tid] = (index < d_nop) ? d_puffs[index].y : -FLT_MAX;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            s_minX[tid] = fminf(s_minX[tid], s_minX[tid + s]);
            s_minY[tid] = fminf(s_minY[tid], s_minY[tid + s]);
            s_maxX[tid] = fmaxf(s_maxX[tid], s_maxX[tid + s]);
            s_maxY[tid] = fmaxf(s_maxY[tid], s_maxY[tid + s]);
        }
        __syncthreads();
    }

    if(tid == 0){
        atomicMinFloat(d_minX, s_minX[0]);
        atomicMinFloat(d_minY, s_minY[0]);
        atomicMaxFloat(d_maxX, s_maxX[0]);
        atomicMaxFloat(d_maxY, s_maxY[0]);
    }
}

// ====================================================================================
// CUDA Kernels - Grid Concentration
// ====================================================================================

/**
 * Accumulate concentration from all puffs at grid points
 * Uses Gaussian puff formula with reflection at ground
 *
 * Thread organization: 1D grid covering all puff-grid combinations
 * Memory access: Atomic operations for concentration accumulation
 *
 * Gaussian puff formula:
 *   C = Q/(2π)^1.5 * σh^2 * σz * exp(-0.5*(dx/σh)^2) * exp(-0.5*(dy/σh)^2)
 *       * (exp(-0.5*(dz/σz)^2) + exp(-0.5*(dz_reflected/σz)^2))
 *
 * @param puffs Array of puff center data
 * @param d_grid Array of grid points
 * @param concs Concentration array to accumulate into
 * @param ngrid Total number of grid points
 */
__global__ void accumulate_conc(
    Gpuff::Puffcenter* puffs,
    RectangleGrid::GridPoint* d_grid,
    float* concs,
    int ngrid)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridIdx = globalIdx % ngrid;
    int puffIdx = globalIdx / ngrid;

    if(puffIdx >= d_nop) return;

    Gpuff::Puffcenter& p = puffs[puffIdx];
    RectangleGrid::GridPoint& g = d_grid[gridIdx];

    if(p.flag){
        float dx = g.x - p.x;
        float dy = g.y - p.y;
        float dz = g.z - p.z;
        float dzv = g.z + p.z;

        if(p.sigma_h != 0.0f && p.sigma_z != 0.0f){
            float contribution = p.conc/(pow(2*PI,1.5)*p.sigma_h*p.sigma_h*p.sigma_z)
                                *exp(-0.5*abs(dx*dx/p.sigma_h/p.sigma_h))
                                *exp(-0.5*abs(dy*dy/p.sigma_h/p.sigma_h))
                                *(exp(-0.5*abs(dz*dz/p.sigma_z/p.sigma_z))
                                +exp(-0.5*abs(dzv*dzv/p.sigma_z/p.sigma_z)));

            atomicAdd(&concs[gridIdx], contribution);
        }
    }
}

/**
 * Validation version of concentration accumulation
 * Same algorithm but may use different grid structure for testing
 *
 * @param puffs Array of puff center data
 * @param d_grid Array of grid points
 * @param concs Concentration array to accumulate into
 * @param ngrid Total number of grid points
 */
__global__ void accumulate_conc_val(
    Gpuff::Puffcenter* puffs,
    RectangleGrid::GridPoint* d_grid,
    float* concs,
    int ngrid)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridIdx = globalIdx % ngrid;
    int puffIdx = globalIdx / ngrid;

    if(puffIdx >= d_nop) return;

    Gpuff::Puffcenter& p = puffs[puffIdx];
    RectangleGrid::GridPoint& g = d_grid[gridIdx];

    if(p.flag){
        float dx = g.x - p.x;
        float dy = g.y - p.y;
        float dz = g.z - p.z;
        float dzv = g.z + p.z;

        if(p.sigma_h != 0.0f && p.sigma_z != 0.0f){
            float contribution = p.conc/(pow(2*PI,1.5)*p.sigma_h*p.sigma_h*p.sigma_z)
                                *exp(-0.5*abs(dx*dx/p.sigma_h/p.sigma_h))
                                *exp(-0.5*abs(dy*dy/p.sigma_h/p.sigma_h))
                                *(exp(-0.5*abs(dz*dz/p.sigma_z/p.sigma_z))
                                +exp(-0.5*abs(dzv*dzv/p.sigma_z/p.sigma_z)));

            atomicAdd(&concs[gridIdx], contribution);
        }

    }
}

// ====================================================================================
// CUDA Kernels - RCAP Receptor Concentration
// ====================================================================================

/**
 * Accumulate concentration at RCAP receptor points
 * Calculates concentration from all puffs at fixed receptor locations
 *
 * Thread organization: 1D grid covering all puff-receptor combinations
 * Memory access: Atomic operations for concentration accumulation
 *
 * @param d_puffs Array of puff center data
 * @param d_receptors Array of receptor points (48 fixed locations)
 */
__global__ void accumulate_conc_RCAP(
    Gpuff::Puffcenter* d_puffs,
    Gpuff::receptors_RCAP* d_receptors)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridIdx = globalIdx % 48;
    int puffIdx = globalIdx / 48;

    if(puffIdx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[puffIdx];
    Gpuff::receptors_RCAP& g = d_receptors[gridIdx];

    g.conc = 0.0;
    float contribution = 0.0;

    // if(p.flag){
    //     float dx = g.x - p.x;
    //     float dy = g.y - p.y;

    //     //printf("dx = %f, dy = %f\n", dx, dy);

    //     if(p.sigma_h != 0.0f){
    //         contribution = p.conc/(pow(2*PI,1.0)*p.sigma_h*p.sigma_h)
    //                             *exp(-0.5*abs(dx*dx/p.sigma_h/p.sigma_h))
    //                             *exp(-0.5*abs(dy*dy/p.sigma_h/p.sigma_h));

    //         atomicAdd(&g.conc, contribution);
    //     }

    //     //printf("%f\n", g.conc);
    // }


    if(p.flag){
        float dx = g.x - p.x;
        float dy = g.y - p.y;
        float dz = g.z - p.z;
        float dzv = g.z + p.z;
        if(p.sigma_h != 0.0f && p.sigma_z != 0.0f){
            float contribution = p.conc/(pow(2*PI,1.5)*p.sigma_h*p.sigma_h*p.sigma_z)
                                *exp(-0.5*abs(dx*dx/p.sigma_h/p.sigma_h))
                                *exp(-0.5*abs(dy*dy/p.sigma_h/p.sigma_h))
                                *(exp(-0.5*abs(dz*dz/p.sigma_z/p.sigma_z))
                                +exp(-0.5*abs(dzv*dzv/p.sigma_z/p.sigma_z)));
            //printf("%e, %e, %e, %e, %e\n", exp(-0.5*abs(dx*dx/p.sigma_h/p.sigma_h)), dx, p.sigma_h, dx/p.sigma_h, contribution);
            atomicAdd(&g.conc, contribution);
        }
    }

}

#endif // GPUFF_KERNELS_CONCENTRATION_CUH