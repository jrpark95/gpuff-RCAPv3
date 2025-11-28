// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel Implementation - Puff Movement and Deposition
// ====================================================================================
//
// File: gpuff_kernels_puff.cuh
// Purpose: Puff transport, deposition, and concentration accumulation kernels
//
// This file contains CUDA kernels for:
//   - Puff activation and flag management
//   - Wind-driven puff transport
//   - Dry deposition processes
//   - Wet scavenging (precipitation washout)
//   - Radioactive decay
//   - Dispersion parameter updates
//   - Concentration grid accumulation
//
// ====================================================================================

#ifndef GPUFF_KERNELS_PUFF_CUH
#define GPUFF_KERNELS_PUFF_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <float.h>
#include "gpuff_struct.cuh"
#include "gpuff_kernels_dispersion.cuh"

// Note: All device constants (d_nop, d_dt, d_etas_hgt_uv, d_etas_hgt_w, d_isPG, d_wc1, d_wc2)
// are defined in gpuff.cuh which is included through the include chain
// No extern declarations needed here
// dimX, dimY, dimZ_pres, dimZ_etas, and invPI are macros defined in gpuff_struct.cuh

// ====================================================================================
// CUDA Kernels - Debug and Utility
// ====================================================================================

/**
 * Debug kernel to check device constant values
 * Prints number of puffs from device constant memory
 * Thread organization: Single thread (0,0) executes
 */
__global__ void checkValueKernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Value of d_nop inside kernel: %d\n", d_nop);
    }
}

/**
 * Debug kernel to print puff time indices
 * Thread organization: 1D grid, one thread per puff
 *
 * @param d_puffs Array of puff center data
 */
__global__ void print_timeidx_kernel(Gpuff::Puffcenter* d_puffs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < d_nop) {
        printf("Timeidx of puff %d: %f\n", tid, d_puffs[tid].y / GRID_SPACING);
    }
}

// ====================================================================================
// CUDA Kernels - Puff Activation
// ====================================================================================

/**
 * Activate puffs based on activation ratio
 * Used for progressive puff release in time-varying simulations
 *
 * Thread organization: 1D grid, one thread per puff
 * Memory access: Coalesced writes to puff flags
 *
 * @param d_puffs Array of puff center data
 * @param activationRatio Fraction of puffs to activate (0.0 to 1.0)
 */
__global__ void update_puff_flags_kernel(
    Gpuff::Puffcenter* d_puffs, float activationRatio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];

    if (idx < int(d_nop * activationRatio)) {
        p.flag = 1;
    }
}

/**
 * Activate puffs based on release time for RCAP simulations
 * Puffs are activated when simulation time exceeds their release time
 *
 * Thread organization: 1D grid, one thread per puff
 * Early exit: Puffs already active are skipped
 *
 * @param d_puffs_RCAP Array of RCAP puff center data
 * @param currentTime Current simulation time (seconds)
 */
__global__ void update_puff_flags2_kernel(
    Gpuff::Puffcenter_RCAP* d_puffs_RCAP, float currentTime)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_nop) return;

    Gpuff::Puffcenter_RCAP& p = d_puffs_RCAP[idx];

    // // Debug output for first few puffs
    // if (idx < 3) {
    //     printf("[PUFF FLAGS DEBUG] Puff %d: releasetime=%.2f, currentTime=%.2f, flag=%d\n",
    //            idx, p.releasetime, currentTime, p.flag);
    // }

    if (p.flag == 1) return;

    if (p.releasetime < currentTime) {
        p.flag = 1;
        // if (idx < 3) {
        //     printf("[PUFF FLAGS DEBUG] Puff %d ACTIVATED at time %.2f\n", idx, currentTime);
        // }
    }
}

// ====================================================================================
// CUDA Kernels - Puff Transport
// ====================================================================================

/**
 * Move puffs by 3D wind field interpolation
 *
 * Performs bilinear interpolation of meteorological wind data (UGRD, VGRD, DZDT)
 * to advect puff positions. Wind components are interpolated from eta-coordinate
 * meteorological grid.
 *
 * Physics:
 *   - Horizontal winds (UGRD, VGRD) on staggered U/V grid
 *   - Vertical motion (DZDT) on W grid
 *   - Bilinear interpolation in horizontal, linear in vertical
 *   - Minimum puff height enforced to prevent ground intersection
 *
 * Thread organization: 1D grid, one thread per puff
 * Memory access: Irregular access pattern due to spatial interpolation
 * Performance note: Memory access pattern not fully coalesced
 *
 * @param d_puffs Array of puff center data
 * @param device_meteorological_data_pres Pressure-level meteorological data
 * @param device_meteorological_data_unis Surface-level meteorological data
 * @param device_meteorological_data_etas Eta-coordinate meteorological data (winds)
 */
__global__ void move_puffs_by_wind_kernel(
    Gpuff::Puffcenter* d_puffs,
    PresData* device_meteorological_data_pres,
    UnisData* device_meteorological_data_unis,
    EtasData* device_meteorological_data_etas)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if (!p.flag) return;

    // Convert puff position to grid indices
    int xidx = int(p.x / GRID_SPACING);
    int yidx = int(p.y / GRID_SPACING);

    // Find vertical level indices for U/V winds and W (vertical motion)
    int zidx_uv = 1;
    int zidx_w = 1;

    for (int i = 0; i < dimZ_etas - 1; i++) {
        if (p.z < d_etas_hgt_uv[i]) {
            zidx_uv = i + 1;
            break;
        }
    }

    for (int i = 0; i < dimZ_etas - 1; i++) {
        if (p.z < d_etas_hgt_w[i]) {
            zidx_w = i + 1;
            break;
        }
    }

    // Validate vertical indices
    if (zidx_uv < 0) {
        printf("Invalid zidx_uv error.\n");
        zidx_uv = 1;
    }

    if (zidx_w < 0) {
        printf("Invalid zidx_w error.\n");
        zidx_w = 1;
    }

    // Calculate interpolation weights (bilinear interpolation)
    float x0 = p.x / GRID_SPACING - xidx;  // Fractional position in cell
    float y0 = p.y / GRID_SPACING - yidx;

    float x1 = 1 - x0;  // Complementary weight
    float y1 = 1 - y0;

    // Bilinear interpolation of U wind component (m/s)
    float xwind = x1 * y1 * device_meteorological_data_etas[xidx * (dimY) * (dimZ_etas) + yidx * (dimZ_etas) + zidx_uv].UGRD +
                  x0 * y1 * device_meteorological_data_etas[(xidx + 1) * (dimY) * (dimZ_etas) + yidx * (dimZ_etas) + zidx_uv].UGRD +
                  x1 * y0 * device_meteorological_data_etas[xidx * (dimY) * (dimZ_etas) + (yidx + 1) * (dimZ_etas) + zidx_uv].UGRD +
                  x0 * y0 * device_meteorological_data_etas[(xidx + 1) * (dimY) * (dimZ_etas) + (yidx + 1) * (dimZ_etas) + zidx_uv].UGRD;

    // Bilinear interpolation of V wind component (m/s)
    float ywind = x1 * y1 * device_meteorological_data_etas[xidx * (dimY) * (dimZ_etas) + yidx * (dimZ_etas) + zidx_uv].VGRD +
                  x0 * y1 * device_meteorological_data_etas[(xidx + 1) * (dimY) * (dimZ_etas) + yidx * (dimZ_etas) + zidx_uv].VGRD +
                  x1 * y0 * device_meteorological_data_etas[xidx * (dimY) * (dimZ_etas) + (yidx + 1) * (dimZ_etas) + zidx_uv].VGRD +
                  x0 * y0 * device_meteorological_data_etas[(xidx + 1) * (dimY) * (dimZ_etas) + (yidx + 1) * (dimZ_etas) + zidx_uv].VGRD;

    // Bilinear interpolation of vertical motion (m/s)
    float zwind = x1 * y1 * device_meteorological_data_etas[xidx * (dimY) * (dimZ_etas) + yidx * (dimZ_etas) + zidx_w].DZDT +
                  x0 * y1 * device_meteorological_data_etas[(xidx + 1) * (dimY) * (dimZ_etas) + yidx * (dimZ_etas) + zidx_w].DZDT +
                  x1 * y0 * device_meteorological_data_etas[xidx * (dimY) * (dimZ_etas) + (yidx + 1) * (dimZ_etas) + zidx_w].DZDT +
                  x0 * y0 * device_meteorological_data_etas[(xidx + 1) * (dimY) * (dimZ_etas) + (yidx + 1) * (dimZ_etas) + zidx_w].DZDT;

    // Update puff position: Forward Euler integration
    p.x += xwind * d_dt;
    p.y += ywind * d_dt;
    p.z += zwind * d_dt;

    // Enforce minimum puff height
    if (p.z < MIN_PUFF_HEIGHT) {
        p.z = MIN_PUFF_HEIGHT;
    }
}

// ====================================================================================
// CUDA Kernels - Deposition and Decay
// ====================================================================================

/**
 * Apply dry deposition to puff concentrations
 *
 * Dry deposition removes material from puffs via gravitational settling
 * and surface impaction. Depletion rate depends on deposition velocity
 * and planetary boundary layer height.
 *
 * Physics:
 *   - Exponential decay: C' = C * exp(-v_d * dt / H_pbl)
 *   - v_d: deposition velocity (m/s), particle/gas specific
 *   - H_pbl: Planetary boundary layer height (m)
 *   - Higher PBL → slower relative depletion
 *
 * Thread organization: 1D grid, one thread per puff
 *
 * @param d_puffs Array of puff center data
 * @param device_meteorological_data_pres Pressure-level meteorological data
 * @param device_meteorological_data_unis Surface-level meteorological data (HPBL)
 * @param device_meteorological_data_etas Eta-coordinate meteorological data
 */
__global__ void dry_deposition_kernel(
    Gpuff::Puffcenter* d_puffs,
    PresData* device_meteorological_data_pres,
    UnisData* device_meteorological_data_unis,
    EtasData* device_meteorological_data_etas)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if (!p.flag) return;

    // Convert puff position to grid indices
    int xidx = int(p.x / GRID_SPACING);
    int yidx = int(p.y / GRID_SPACING);

    // Calculate interpolation weights
    float x0 = p.x / GRID_SPACING - xidx;
    float y0 = p.y / GRID_SPACING - yidx;
    float x1 = 1 - x0;
    float y1 = 1 - y0;

    // Bilinear interpolation of planetary boundary layer height (meters)
    float mixing_height = x1 * y1 * device_meteorological_data_unis[xidx * (dimY) + yidx].HPBL +
                          x0 * y1 * device_meteorological_data_unis[(xidx + 1) * (dimY) + yidx].HPBL +
                          x1 * y0 * device_meteorological_data_unis[xidx * (dimY) + (yidx + 1)].HPBL +
                          x0 * y0 * device_meteorological_data_unis[(xidx + 1) * (dimY) + (yidx + 1)].HPBL;

    // Apply dry deposition: exponential decay
    p.conc *= exp(-p.drydep_vel * d_dt / mixing_height);
}

/**
 * Apply wet scavenging (washout) to puff concentrations
 *
 * Wet scavenging removes material via precipitation scavenging.
 * Only active when relative humidity exceeds threshold (80%).
 *
 * Physics:
 *   - Washout coefficient: Lambda = 3.5e-5 * (RH - 0.8) / 0.2 [s^-1]
 *   - Only active when RH > 80%
 *   - Linear dependence on relative humidity above threshold
 *   - Exponential decay: C' = C * exp(-Lambda * dt)
 *
 * Reference: Simplified washout model from EPA regulatory guidance
 *
 * Thread organization: 1D grid, one thread per puff
 *
 * @param d_puffs Array of puff center data
 * @param device_meteorological_data_pres Pressure-level meteorological data (RH)
 * @param device_meteorological_data_unis Surface-level meteorological data
 * @param device_meteorological_data_etas Eta-coordinate meteorological data
 */
__global__ void wet_scavenging_kernel(
    Gpuff::Puffcenter* d_puffs,
    PresData* device_meteorological_data_pres,
    UnisData* device_meteorological_data_unis,
    EtasData* device_meteorological_data_etas)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if (!p.flag) return;

    // Convert puff position to grid indices
    int xidx = int(p.x / GRID_SPACING);
    int yidx = int(p.y / GRID_SPACING);

    // Find vertical pressure level index
    int zidx_pres = 1;
    for (int i = 0; i < dimZ_pres - 1; i++) {
        if (p.z < device_meteorological_data_pres[xidx * (dimY) * (dimZ_pres) + yidx * (dimZ_pres) + i].HGT) {
            zidx_pres = i + 1;
            break;
        }
    }

    if (zidx_pres < 0) {
        printf("Invalid zidx_pres error.\n");
        zidx_pres = 1;
    }

    // Calculate interpolation weights
    float x0 = p.x / GRID_SPACING - xidx;
    float y0 = p.y / GRID_SPACING - yidx;
    float x1 = 1 - x0;
    float y1 = 1 - y0;

    // Bilinear interpolation of relative humidity (0-1)
    float relative_humidity = x1 * y1 * device_meteorological_data_pres[xidx * (dimY) * (dimZ_pres) + yidx * (dimZ_pres) + zidx_pres].RH +
                              x0 * y1 * device_meteorological_data_pres[(xidx + 1) * (dimY) * (dimZ_pres) + yidx * (dimZ_pres) + zidx_pres].RH +
                              x1 * y0 * device_meteorological_data_pres[xidx * (dimY) * (dimZ_pres) + (yidx + 1) * (dimZ_pres) + zidx_pres].RH +
                              x0 * y0 * device_meteorological_data_pres[(xidx + 1) * (dimY) * (dimZ_pres) + (yidx + 1) * (dimZ_pres) + zidx_pres].RH;

    // Calculate washout coefficient (only active above threshold)
    if (relative_humidity > WET_SCAVENGING_RH_THRESHOLD) {
        float lambda = WET_SCAVENGING_LAMBDA_COEFF * (relative_humidity - WET_SCAVENGING_RH_THRESHOLD) /
                       (1.0f - WET_SCAVENGING_RH_THRESHOLD);

        // Apply wet scavenging: exponential decay
        p.conc *= exp(-lambda * d_dt);
    }
}

/**
 * Apply radioactive decay to puff concentrations
 *
 * Updates puff concentration due to radioactive decay.
 * Each puff carries its decay constant based on nuclide half-life.
 *
 * Physics:
 *   - Exponential decay: C' = C * exp(-lambda * dt)
 *   - lambda = ln(2) / t_half
 *   - Decay constant stored in puff structure
 *
 * Thread organization: 1D grid, one thread per puff
 * Memory access: Fully coalesced
 *
 * @param d_puffs Array of puff center data
 * @param device_meteorological_data_pres Pressure-level meteorological data (unused)
 * @param device_meteorological_data_unis Surface-level meteorological data (unused)
 * @param device_meteorological_data_etas Eta-coordinate meteorological data (unused)
 */
__global__ void radioactive_decay_kernel(
    Gpuff::Puffcenter* d_puffs,
    PresData* device_meteorological_data_pres,
    UnisData* device_meteorological_data_unis,
    EtasData* device_meteorological_data_etas)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if (!p.flag) return;

    // Apply radioactive decay: exponential decay
    p.conc *= exp(-p.decay_const * d_dt);
}

__global__ void puff_dispersion_update(
    Gpuff::Puffcenter* d_puffs,
    PresData* device_meteorological_data_pres,
    UnisData* device_meteorological_data_unis,
    EtasData* device_meteorological_data_etas)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    int xidx = int(p.x/1500.0);
    int yidx = int(p.y/1500.0);
    int zidx_uv = 1;
    int zidx_w = 1;

    for(int i=0; i<dimZ_etas-1; i++){
        if(p.z<d_etas_hgt_uv[i]){
            zidx_uv=i+1;
            break;
        }
    }

    for(int i=0; i<dimZ_etas-1; i++){
        if(p.z<d_etas_hgt_w[i]){
            zidx_w=i+1;
            break;
        }
    }

    if(zidx_uv<0) {
        printf("Invalid zidx_uv error.\n");
        zidx_uv = 1;
    }

    if(zidx_w<0) {
        printf("Invalid zidx_w error.\n");
        zidx_w = 1;
    }

    float x0 = p.x/1500.0-xidx;
    float y0 = p.y/1500.0-yidx;

    float x1 = 1-x0;
    float y1 = 1-y0;

    float xwind = x1*y1*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].UGRD
                    +x0*y1*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].UGRD
                    +x1*y0*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].UGRD
                    +x0*y0*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].UGRD;

    float ywind = x1*y1*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].VGRD
                    +x0*y1*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].VGRD
                    +x1*y0*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].VGRD
                    +x0*y0*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].VGRD;

    float zwind = x1*y1*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_w].DZDT
                    +x0*y1*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_w].DZDT
                    +x1*y0*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_w].DZDT
                    +x0*y0*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_w].DZDT;


    float vel = sqrt(xwind*xwind + ywind*ywind + zwind*zwind);

    //printf("zwind: %f, vel: %f ", zwind, vel);

    float t0 = x1*y1*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + yidx*(dimZ_pres)].TMP
                +x0*y1*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + yidx*(dimZ_pres)].TMP
                +x1*y0*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres)].TMP
                +x0*y0*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres)].TMP;

    float tu = x1*y1*device_meteorological_data_unis[xidx*(dimY) + yidx].T1P5
                +x0*y1*device_meteorological_data_unis[(xidx+1)*(dimY) + yidx].T1P5
                +x1*y0*device_meteorological_data_unis[xidx*(dimY) + (yidx+1)].T1P5
                +x0*y0*device_meteorological_data_unis[(xidx+1)*(dimY) + (yidx+1)].T1P5;

    float gph0 = x1*y1*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + yidx*(dimZ_pres)].HGT
                +x0*y1*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + yidx*(dimZ_pres)].HGT
                +x1*y0*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres)].HGT
                +x0*y0*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres)].HGT;

    float dtp100m = 100.0*(t0-tu)/(gph0-1.5);

    int PasquillCategory = 0;

    if(dtp100m < -1.9) PasquillCategory = 0;        // A: Extremely unstable
    else if(dtp100m < -1.7) PasquillCategory = 1;   // B: Moderately unstable
    else if(dtp100m < -1.5) PasquillCategory = 2;   // C: Slightly unstable
    else if(dtp100m < -0.5) PasquillCategory = 3;   // D: Neutral
    else if(dtp100m < 1.5) PasquillCategory = 4;    // E: Slightly stable
    else if(dtp100m < 4.0) PasquillCategory = 5;    // F: Moderately stable
    else PasquillCategory = 6;                      // G: Extremely stable

    float new_virtual_distance_h = NewtonRaphson_h(PasquillCategory, p.sigma_h, p.virtual_distance) + vel*d_dt;
    float new_virtual_distance_z = NewtonRaphson_z(PasquillCategory, p.sigma_z, p.virtual_distance) + vel*d_dt;

    if(d_isPG){
        p.sigma_h = Sigma_h_Pasquill_Gifford(PasquillCategory, new_virtual_distance_h);
        p.sigma_z = Sigma_z_Pasquill_Gifford(PasquillCategory, new_virtual_distance_z);
    }
    else{
        //p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_z);
    }

    p.virtual_distance = new_virtual_distance_h;

}

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

// Validation kernels for testing
__global__ void move_puffs_by_wind_val(Gpuff::Puffcenter* d_puffs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    float xwind = 1.0f;
    float ywind = 0.0f;
    float zwind = 0.0f;

    p.x += xwind*d_dt;
    p.y += ywind*d_dt;
    p.z += zwind*d_dt;

    if(p.z<0.0) p.z=-p.z;
}

__global__ void dry_deposition_val(Gpuff::Puffcenter* d_puffs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    p.conc *= exp(-p.drydep_vel*d_dt/1000.0);
}

__global__ void wet_scavenging_val(Gpuff::Puffcenter* d_puffs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    float Lambda = 3.5e-5*(1.0-0.8)/(1.0-0.8);
    p.conc *= exp(-Lambda*d_dt);
}

__global__ void radioactive_decay_val(Gpuff::Puffcenter* d_puffs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    p.conc *= exp(-p.decay_const*d_dt);
}

__global__ void puff_dispersion_update_val(Gpuff::Puffcenter* d_puffs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    float xwind = 1.0f;
    float ywind = 0.0f;
    float zwind = 0.0f;

    float vel = sqrt(xwind*xwind + ywind*ywind + zwind+zwind);
    int PasquillCategory = 5;

    float new_virtual_distance_h = NewtonRaphson_h(PasquillCategory, p.sigma_h, p.virtual_distance) + vel*d_dt;
    float new_virtual_distance_z = NewtonRaphson_z(PasquillCategory, p.sigma_z, p.virtual_distance) + vel*d_dt;

    if(d_isPG){
        p.sigma_h = Sigma_h_Pasquill_Gifford(PasquillCategory, new_virtual_distance_h);
        p.sigma_z = Sigma_z_Pasquill_Gifford(PasquillCategory, new_virtual_distance_z);
    }
    else{
        //p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_z);
    }

    p.virtual_distance = new_virtual_distance_h;
}

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

// RCAP specific kernels
__global__ void move_puffs_by_wind_RCAP(
    Gpuff::Puffcenter* d_puffs,
    float* d_RCAP_windir,
    float* d_RCAP_winvel,
    float* d_radi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    float xwind = p.windvel*cos(p.windir);
    float ywind = p.windvel*sin(p.windir);

    p.x += xwind*d_dt;
    p.y += ywind*d_dt;
}

__global__ void move_puffs_by_wind_RCAP2(
    Gpuff::Puffcenter_RCAP* d_puffs_RCAP, float* d_Vdepo, float* d_particleSizeDistr,
    int EP_endRing, float* d_ground_deposit, NuclideData* d_ND, float* d_radius, int numRad, int numTheta)
{
    int puffidx = threadIdx.x;
    int simidx = blockIdx.x;

    int idx = simidx * blockDim.x + puffidx;
    if (idx >= d_nop) return;

    Gpuff::Puffcenter_RCAP& p = d_puffs_RCAP[idx];
    if (p.flag == 0) return;

    float xwind = p.windvel * cos(p.windir);
    float ywind = p.windvel * sin(p.windir);

    p.x += xwind * d_dt;
    p.y += ywind * d_dt;

    // Disabled: 5km boundary deactivation
    // if (p.x * p.x + p.y * p.y > 5000.0 * 5000.0) {
    //     p.flag = 0;
    //     return;
    // }

    p.virtual_distance += p.windvel * d_dt;

    if (d_isPG) {
        p.sigma_h = Sigma_h_Pasquill_Gifford(p.stab - 1, p.virtual_distance);
        p.sigma_z = Sigma_z_Pasquill_Gifford(p.stab - 1, p.virtual_distance);
    }
    else {
        //p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_z);
    }

    float wetf = expf(-d_wc1*powf(p.rain, d_wc2)*d_dt);

    float r = sqrt(p.x * p.x + p.y * p.y);
    float theta = atan2(p.y, p.x);

    int rad_idx = 0;
    for (int i = 0; i < numRad; ++i) {
        if (r < d_radius[i]) {
            rad_idx = i;
            break;
        }
    }
    int theta_idx = round(p.windir * 8.0f * invPI);
    theta_idx = (theta_idx - 1) % 16 + 1;

    for (int nuc_idx = 0; nuc_idx < MAX_NUCLIDES; ++nuc_idx) {
        NuclideData nuclide = d_ND[nuc_idx];

        float decay_factor = expf(-logf(2.0f) / nuclide.half_life * d_dt);

        int group = nuclide.chemical_group;
        if (group < 1 || group > ELEMENT_COUNT) {
            continue;
        }

        if (nuclide.dry_deposition == true) {
            float conc = p.conc[nuc_idx];
            float f_total = 0.0f;

            for (int size_idx = 0; size_idx < PARTICLE_COUNT; ++size_idx) {
                float fraction = d_particleSizeDistr[p.unitidx * (ELEMENT_COUNT * PARTICLE_COUNT)
                                                    + (group - 1) * PARTICLE_COUNT + size_idx];
                float vdep = d_Vdepo[size_idx];
                float f_size = expf(-vdep * d_dt / 1500.0f);
                f_total += fraction * f_size;
            }

            float new_conc = nuclide.wet_deposition ? conc * f_total * wetf : conc * f_total;
            p.conc[nuc_idx] = new_conc * decay_factor;

            float deposition = conc - new_conc;
        }
    }
}

// CPU version of update_puff_flags2 for validation
void Gpuff::update_puff_flags2_cpu(float currentTime, int nop) {
    for (int idx = 0; idx < nop; ++idx) {
        Gpuff::Puffcenter_RCAP& p = puffs_RCAP[idx];
        if (p.flag == 1) continue;

        if (p.releasetime < currentTime) {
            p.flag = 1;
        }
    }
}

// CPU version of move_puffs_by_wind_RCAP2 for validation
void Gpuff::move_puffs_by_wind_RCAP2_cpu(
    int EP_endRing, std::vector<NuclideData> ND, float* radius,
    int numRad, int numTheta, int nop)
{
    for (int idx = 0; idx < nop; ++idx) {
        Gpuff::Puffcenter_RCAP& p = puffs_RCAP[idx];
        if (p.flag == 0) continue;

        float xwind = p.windvel * cos(p.windir);
        float ywind = p.windvel * sin(p.windir);

        p.x += xwind * dt;
        p.y += ywind * dt;

        p.virtual_distance += p.windvel * dt;

        if (1) {
            p.sigma_h = Sigma_h_Pasquill_Gifford_cpu(p.stab - 1, p.virtual_distance);
            p.sigma_z = Sigma_z_Pasquill_Gifford_cpu(p.stab - 1, p.virtual_distance);
        }
        else {

        }

        float wetf = expf(-wc1 * powf(p.rain, wc2) * dt);

        float r = sqrt(p.x * p.x + p.y * p.y);
        float theta = atan2(p.y, p.x);

        int rad_idx = 0;
        for (int i = 0; i < numRad; ++i) {
            if (r < radius[i]) {
                rad_idx = i;
                break;
            }
        }
        int theta_idx = round(p.windir * 8.0f * (1 / PI));
        theta_idx = (theta_idx - 1) % 16 + 1;

        for (int nuc_idx = 0; nuc_idx < MAX_NUCLIDES; ++nuc_idx) {
            NuclideData nuclide = ND[nuc_idx];

            float decay_factor = expf(-logf(2.0f) / nuclide.half_life * dt);

            int group = nuclide.chemical_group;
            if (group < 1 || group > ELEMENT_COUNT) {
                continue;
            }

            if (nuclide.dry_deposition) {
                float conc = p.conc[nuc_idx];
                float f_total = 0.0f;

                for (int size_idx = 0; size_idx < PARTICLE_COUNT; ++size_idx) {
                    float fraction = particleSizeDistr[0][(group - 1)][size_idx];
                    float vdep = Vdepo[size_idx];
                    float f_size = expf(-vdep * dt / 1500.0f);
                    f_total += fraction * f_size;
                }

                float new_conc = nuclide.wet_deposition ? conc * f_total * wetf : conc * f_total;
                p.conc[nuc_idx] = new_conc * decay_factor;

                float deposition = conc - new_conc;
            }
        }
    }
}

#endif // GPUFF_KERNELS_PUFF_CUH