// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel Implementation - Evacuation and RCAP Specific
// ====================================================================================
//
// File: gpuff_kernels_evacuation.cuh
// Purpose: Evacuation simulation and RCAP-specific transport kernels
//
// This file contains CUDA kernels for:
//   - Evacuee movement and routing
//   - Time-in/time-out calculations for puffs
//   - RCAP-specific puff transport and dispersion
//   - Ground deposition tracking
//   - Concentration accumulation for RCAP receptors
//
// ====================================================================================

#ifndef GPUFF_KERNELS_EVACUATION_CUH
#define GPUFF_KERNELS_EVACUATION_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "gpuff_struct.cuh"
#include "gpuff_kernels_dispersion.cuh"

// Note: All device constants (d_nop, d_dt, d_etas_hgt_uv, d_etas_hgt_w, d_isPG, d_wc1, d_wc2)
// are defined in gpuff.cuh which is included through the include chain
// No extern declarations needed here
// dimX, dimY, dimZ_pres, dimZ_etas, and invPI are macros defined in gpuff_struct.cuh

// Evacuation direction constants
#define DIR_NONE 0
#define DIR_F 1  // Forward
#define DIR_B 2  // Backward
#define DIR_L 3  // Left
#define DIR_R 4  // Right

// ====================================================================================
// Evacuation Calculation Kernels
// ====================================================================================

/**
 * 1D evacuation calculation kernel
 * Simulates evacuee movement based on evacuation plan parameters
 *
 * @param d_puffs_RCAP RCAP puff centers
 * @param d_dir Evacuation direction grid
 * @param d_evacuee Evacuee array
 * @param d_radius Radial grid boundaries
 * @param numRad Number of radial zones
 * @param numTheta Number of angular sectors
 * @param dnop Number of evacuees
 * @param evaEndRing Evacuation end ring
 * @param EP_endRing Emergency planning end ring
 * @param d_ground_deposit Ground deposition array
 * @param dEP Evacuation data
 * @param currentTime Current simulation time
 */
__global__ void evacuation_calculation_1D(
    Gpuff::Puffcenter_RCAP* d_puffs_RCAP, int* d_dir, Evacuee* d_evacuee,
    float* d_radius, int numRad, int numTheta, int dnop,
    int evaEndRing, int EP_endRing, float* d_ground_deposit, const EvacuationData* dEP, float currentTime)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= dnop) {
        return;
    }

    Evacuee& p = d_evacuee[idx];

    if (p.r >= d_radius[EP_endRing - 1]) {
        p.speed = 0.0f;
        return;
    }

    int rad_idx = 0;
    for (int i = 0; i < numRad; ++i) {
        if (p.r <= d_radius[i]) {
            rad_idx = i;
            break;
        }
    }

    if (p.rad0 >= dEP->evaEndRing) return;

    // Check shelter delay phase
    if (currentTime < dEP->alarmTime + dEP->shelterDelay[p.rad0]) {
        p.speed = 0.0f;
        p.flag = 0;
    }
    // Check shelter duration phase
    else if (currentTime < dEP->alarmTime + dEP->shelterDelay[p.rad0] + dEP->shelterDuration[p.rad0]) {
        p.speed = 0.0f;
        p.flag = 1;
    }
    // Evacuation phase with time-varying speeds
    else {
        for (int i = 0; i < dEP->nSpeedPeriod-1; i++) {
            float sum_durations = 0.0f;
            for (int j = 0; j < i+1; j++) sum_durations += dEP->durations[j];
            if (currentTime < dEP->alarmTime + dEP->shelterDelay[p.rad0] + dEP->shelterDuration[p.rad0]
                + sum_durations) {
                p.speed = dEP->speeds[i];
                p.flag = 2;
                break;
            }
        }
        float sum_durations = 0.0f;
        for (int j = 0; j < dEP->nSpeedPeriod-1; j++) sum_durations += dEP->durations[j];
        if (currentTime > dEP->alarmTime + dEP->shelterDelay[p.rad0] + dEP->shelterDuration[p.rad0]
            + sum_durations) {
            p.speed = dEP->speeds[dEP->nSpeedPeriod-1];
            p.flag = 2;
        }
    }

    int theta_idx = static_cast<int>(p.theta / (2 * PI / numTheta)) % numTheta;
    int dir = d_dir[rad_idx * numTheta + theta_idx];

    switch (dir) {
    case DIR_F: // Forward
        p.r += p.speed * d_dt;
        break;
    case DIR_B: // Backward
        p.r -= p.speed * d_dt;
        if (p.r < 0) p.r = 0;
        break;
    case DIR_L: // Left
        p.theta -= p.speed * d_dt / p.r;
        if (p.theta < 0) p.theta += 2 * PI;
        break;
    case DIR_R: // Right
        p.theta += p.speed * d_dt / p.r;
        if (p.theta >= 2 * PI) p.theta -= 2 * PI;
        break;
    default: // DIR_NONE or any other invalid direction
        // No movement
        break;
    }
}

/**
 * 2D evacuation calculation kernel for multiple simulations
 * Uses 2D grid to handle multiple simulations in parallel
 *
 * @param d_puffs_RCAP RCAP puff centers
 * @param d_dir Evacuation direction grid
 * @param d_evacuee Evacuee array
 * @param d_radius Radial grid boundaries
 * @param numRad Number of radial zones
 * @param numTheta Number of angular sectors
 * @param dnop Number of evacuees per simulation
 * @param evaEndRing Evacuation end ring
 * @param EP_endRing Emergency planning end ring
 * @param d_ground_deposit Ground deposition array
 */
__global__ void evacuation_calculation_2D(
    Gpuff::Puffcenter_RCAP* d_puffs_RCAP, int* d_dir, Evacuee* d_evacuee,
    float* d_radius, int numRad, int numTheta, int dnop,
    int evaEndRing, int EP_endRing, float* d_ground_deposit)
{
    int sim_idx = blockIdx.x;  // Simulation index
    int evac_idx = threadIdx.x + blockIdx.y * blockDim.x;  // Evacuee index within the simulation

    if (evac_idx >= 98) {
        return;
    }

    int global_evac_idx = sim_idx * 97 + evac_idx;
    Evacuee& p = d_evacuee[global_evac_idx];

    if (p.flag == 0) {
        return;
    }

    if (p.r >= d_radius[EP_endRing - 1]) {
        p.speed = 0.0f;
        return;
    }

    int rad_idx = 0;
    for (int i = 0; i < numRad; ++i) {
        if (p.r < d_radius[i]) {
            rad_idx = i;
            break;
        }
    }

    int theta_idx = static_cast<int>(p.theta / (2 * PI / numTheta)) % numTheta;
    int dir = d_dir[rad_idx * numTheta + theta_idx];

    switch (dir) {
    case DIR_F: // Forward
        p.r += p.speed * d_dt;
        break;
    case DIR_B: // Backward
        p.r -= p.speed * d_dt;
        if (p.r < 0) p.r = 0;
        break;
    case DIR_L: // Left
        p.theta -= p.speed * d_dt / p.r;
        if (p.theta < 0) p.theta += 2 * PI;
        break;
    case DIR_R: // Right
        p.theta += p.speed * d_dt / p.r;
        if (p.theta >= 2 * PI) p.theta -= 2 * PI;
        break;
    default: // DIR_NONE or any other invalid direction
        // No movement
        break;
    }
}

// ====================================================================================
// RCAP Specific Transport Kernels
// ====================================================================================

/**
 * Track time-in/time-out for puffs crossing radial boundaries
 * Calculates fallout fractions for each ring crossed
 *
 * @param d_puffs Puff centers
 * @param d_RCAP_windir Wind direction array
 * @param d_RCAP_winvel Wind velocity array
 * @param d_radi Radial boundaries
 * @param currentTime Current simulation time
 * @param d_size Particle size distribution
 * @param d_vdepo Deposition velocities
 */
__global__ void time_inout_RCAP(
    Gpuff::Puffcenter* d_puffs,
    float* d_RCAP_windir,
    float* d_RCAP_winvel,
    float* d_radi,
    float currentTime,
    float** d_size,
    float* d_vdepo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    if(p.tin[0]<1.0e-8) {
        p.tin[0] = currentTime;
    }

    float hdist = 1.0*d_radi[p.head_radidx] + 0.0*d_radi[p.head_radidx+1]
                   - sqrt(p.x*p.x+p.y*p.y) - p.sigma_h*2.15;
    float tdist = - sqrt(p.x*p.x+p.y*p.y) + 1.0*d_radi[p.tail_radidx]
                    + 0.0*d_radi[p.tail_radidx+1] + p.sigma_h*2.15;

    // Check if puff head crosses boundary
    if (hdist * p.head_dist < 0) {
        if(idx==1) {
            printf("hdist = %f, head_dist = %f\n", hdist, p.head_dist);
            printf("head_radidx = %d, currentTime = %f\n", p.head_radidx, currentTime);
        }
        p.tin[p.head_radidx+1] = currentTime;
        p.head_radidx++;
        hdist = 0.0;
    }

    // Check if puff tail crosses boundary
    if (tdist * p.tail_dist < 0) {
        if(idx==10) {
            printf("tdist = %f, tail_dist = %f\n", tdist, p.tail_dist);
            printf("tail_radidx = %d, currentTime = %f\n", p.tail_radidx, currentTime);
        }
        p.tout[p.tail_radidx] = currentTime;

        tdist = 0.0;

        float fd_temp = 1.0;
        float H = 1000.0;
        float rain = 1.0;

        float C1 = 1.89e-5;
        float C2 = 0.664;

        // Calculate fallout for each nuclide and particle size
        for (int iNuclide = 0; iNuclide < 9; iNuclide++) {
            if (iNuclide == 0) continue;
            p.fw[iNuclide][p.tail_radidx] = exp(-C1 * pow(rain, C2) * (p.tout[p.tail_radidx] - p.tin[p.tail_radidx]));

            for (int iSize = 0; iSize < 10; iSize++) {
                p.fd[iNuclide][p.tail_radidx] = exp(-d_vdepo[iSize] * (p.tout[p.tail_radidx] - p.tin[p.tail_radidx]) / H);
                p.fallout[iNuclide][p.tail_radidx] += d_size[iNuclide][iSize] * (1 - p.fd[iNuclide][p.tail_radidx] * p.fw[iNuclide][p.tail_radidx]);
            }
            p.conc_arr[iNuclide] = p.conc_arr[iNuclide] * (1 - p.fallout[iNuclide][p.tail_radidx]);
        }
        p.tail_radidx++;
    }

    float xwind = p.windvel*cos(p.windir);
    float ywind = p.windvel*sin(p.windir);

    p.x += xwind*d_dt;
    p.y += ywind*d_dt;

    p.head_dist = hdist;
    p.tail_dist = tdist;
}

/**
 * Update dispersion parameters for RCAP puffs
 *
 * @param d_puffs Puff centers
 * @param d_RCAP_windir Wind direction array
 * @param d_RCAP_winvel Wind velocity array
 * @param d_radi Radial boundaries
 */
__global__ void puff_dispersion_update_RCAP(
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

    float vel = sqrt(xwind*xwind + ywind*ywind);

    int PasquillCategory = p.stab-1;

    float new_virtual_distance_h = NewtonRaphson_h(PasquillCategory, p.sigma_h, p.virtual_distance) + vel*d_dt;

    if(d_isPG){
        p.sigma_h = Sigma_h_Pasquill_Gifford(PasquillCategory, new_virtual_distance_h);
    }
    else{
        //p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
    }

    p.virtual_distance = new_virtual_distance_h;
}

/**
 * Alternative dispersion update for RCAP puffs
 *
 * @param d_puffs Puff centers
 * @param d_RCAP_windir Wind direction array
 * @param d_RCAP_winvel Wind velocity array
 */
__global__ void puff_dispersion_update_RCAP2(
    Gpuff::Puffcenter* d_puffs,
    float* d_RCAP_windir,
    float* d_RCAP_winvel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if (!p.flag) return;

    float xwind = p.windvel * cos(p.windir);
    float ywind = p.windvel * sin(p.windir);

    float vel = sqrt(xwind * xwind + ywind * ywind);

    int PasquillCategory = p.stab - 1;

    float new_virtual_distance_h = NewtonRaphson_h(PasquillCategory, p.sigma_h, p.virtual_distance) + vel * d_dt;

    if (d_isPG) {
        p.sigma_h = Sigma_h_Pasquill_Gifford(PasquillCategory, new_virtual_distance_h);
    }
    else {
        //p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
    }

    p.virtual_distance = new_virtual_distance_h;
}

/**
 * Accumulate concentration at RCAP receptor locations
 * Uses Gaussian puff formula with reflection at ground
 *
 * @param d_puffs Puff centers
 * @param d_receptors RCAP receptor locations
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
            atomicAdd(&g.conc, contribution);
        }
    }
}

/**
 * Move puffs by variable wind field
 * Includes bounds checking and interpolation of meteorological data
 *
 * @param d_puffs Puff centers
 * @param device_meteorological_data_pres Pressure level data
 * @param device_meteorological_data_unis Surface data
 * @param device_meteorological_data_etas Eta coordinate data
 * @param currentTime Current simulation time
 */
__global__ void move_puffs_by_wind_var(
    Gpuff::Puffcenter* d_puffs,
    PresData* device_meteorological_data_pres,
    UnisData* device_meteorological_data_unis,
    EtasData* device_meteorological_data_etas,
    float currentTime)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if (!p.flag) return;

    int xidx = int(p.x / 1500.0);
    int yidx = int(p.y / 1500.0);
    int zidx_uv = 1;
    int zidx_w = 1;

    if (xidx < 2 || xidx > dimX-2) {
        printf("xidx error! xidx = %d\n", xidx);
    }

    if (yidx < 2 || yidx > dimY - 2) {
        printf("yidx error! yidx = %d\n", yidx);
    }

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

    if (zidx_uv < 0) {
        printf("Invalid zidx_uv error.\n");
        zidx_uv = 1;
    }

    if (zidx_w < 0) {
        printf("Invalid zidx_w error.\n");
        zidx_w = 1;
    }

    float x0 = p.x / 1500.0 - xidx;
    float y0 = p.y / 1500.0 - yidx;

    float x1 = 1 - x0;
    float y1 = 1 - y0;

    // Bilinear interpolation of wind components
    float xwind = x1 * y1 * device_meteorological_data_etas[xidx * (dimY) * (dimZ_etas)+yidx * (dimZ_etas)+zidx_uv].UGRD
        + x0 * y1 * device_meteorological_data_etas[(xidx + 1) * (dimY) * (dimZ_etas)+yidx * (dimZ_etas)+zidx_uv].UGRD
        + x1 * y0 * device_meteorological_data_etas[xidx * (dimY) * (dimZ_etas)+(yidx + 1) * (dimZ_etas)+zidx_uv].UGRD
        + x0 * y0 * device_meteorological_data_etas[(xidx + 1) * (dimY) * (dimZ_etas)+(yidx + 1) * (dimZ_etas)+zidx_uv].UGRD;

    float ywind = x1 * y1 * device_meteorological_data_etas[xidx * (dimY) * (dimZ_etas)+yidx * (dimZ_etas)+zidx_uv].VGRD
        + x0 * y1 * device_meteorological_data_etas[(xidx + 1) * (dimY) * (dimZ_etas)+yidx * (dimZ_etas)+zidx_uv].VGRD
        + x1 * y0 * device_meteorological_data_etas[xidx * (dimY) * (dimZ_etas)+(yidx + 1) * (dimZ_etas)+zidx_uv].VGRD
        + x0 * y0 * device_meteorological_data_etas[(xidx + 1) * (dimY) * (dimZ_etas)+(yidx + 1) * (dimZ_etas)+zidx_uv].VGRD;

    float zwind = x1 * y1 * device_meteorological_data_etas[xidx * (dimY) * (dimZ_etas)+yidx * (dimZ_etas)+zidx_w].DZDT
        + x0 * y1 * device_meteorological_data_etas[(xidx + 1) * (dimY) * (dimZ_etas)+yidx * (dimZ_etas)+zidx_w].DZDT
        + x1 * y0 * device_meteorological_data_etas[xidx * (dimY) * (dimZ_etas)+(yidx + 1) * (dimZ_etas)+zidx_w].DZDT
        + x0 * y0 * device_meteorological_data_etas[(xidx + 1) * (dimY) * (dimZ_etas)+(yidx + 1) * (dimZ_etas)+zidx_w].DZDT;

    p.x += xwind * d_dt;
    p.y += ywind * d_dt;
    p.z += zwind * d_dt;

    if (p.z < 2.0) p.z = 2.0;
}

// ====================================================================================
// CPU Validation Functions
// ====================================================================================

/**
 * CPU version of evacuation calculation for validation
 */
void evacuation_calculation_cpu(
    EvacuationDirections& ED, std::vector<Evacuee>& evacuee,
    float* radius, int numRad, int numTheta, int nop,
    int evaEndRing, int EP_endRing)
{
    for (int idx = 0; idx < nop; ++idx) {
        Evacuee& p = evacuee[idx];

        if (p.flag == 0) {
            continue;
        }

        if (p.r >= radius[EP_endRing - 1]) {
            p.speed = 0.0f;
            continue;
        }

        int rad_idx = 0;
        for (int i = 0; i < numRad; ++i) {
            if (p.r < radius[i]) {
                rad_idx = i;
                break;
            }
        }

        int theta_idx = static_cast<int>(p.theta / (2 * PI / numTheta)) % numTheta;
        int direction = ED.get(rad_idx, theta_idx);

        switch (direction) {
        case DIR_F: // Forward
            p.r += p.speed * dt;
            break;
        case DIR_B: // Backward
            p.r -= p.speed * dt;
            if (p.r < 0) p.r = 0;
            break;
        case DIR_L: // Left
            p.theta -= p.speed * dt / p.r;
            if (p.theta < 0) p.theta += 2 * PI;
            break;
        case DIR_R: // Right
            p.theta += p.speed * dt / p.r;
            if (p.theta >= 2 * PI) p.theta -= 2 * PI;
            break;
        default: // DIR_NONE or any other invalid direction
            // No movement
            break;
        }
    }
}

#endif // GPUFF_KERNELS_EVACUATION_CUH