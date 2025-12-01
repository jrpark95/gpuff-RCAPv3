// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel - Evacuation Calculations
// ====================================================================================
//
// File: gpuff_kernels_evacuation.cuh
// Purpose: GPU kernels for evacuee movement and shelter calculations
//
// This file contains:
//   - Evacuee movement in polar coordinates
//   - Shelter/evacuation phase management
//   - Speed calculations based on time periods
//   - Direction-based movement (Forward/Backward/Left/Right)
//
// ====================================================================================

#ifndef GPUFF_KERNELS_EVACUATION_CUH
#define GPUFF_KERNELS_EVACUATION_CUH

#include "gpuff_kernels_constants.cuh"

// Direction constants for evacuation
#define DIR_NONE 0
#define DIR_F 1  // Forward (radially outward)
#define DIR_B 2  // Backward (radially inward)
#define DIR_L 3  // Left (counter-clockwise)
#define DIR_R 4  // Right (clockwise)

// ====================================================================================
// CUDA Kernels - 1D Evacuation
// ====================================================================================

/**
 * 1D evacuation calculation kernel
 * Updates evacuee positions based on current phase and direction
 *
 * Evacuation phases:
 *   1. Pre-alarm: No movement
 *   2. Shelter delay: Waiting period before sheltering
 *   3. Shelter: In shelter (protected)
 *   4. Evacuation: Moving according to speed profile
 *
 * @param d_puffs_RCAP Puff data (unused in movement)
 * @param d_dir Direction array [radial][angular]
 * @param d_evacuee Evacuee position/state data
 * @param d_radius Radial ring distances
 * @param numRad Number of radial rings
 * @param numTheta Number of angular sectors
 * @param dnop Number of evacuees
 * @param evaEndRing Evacuation end ring
 * @param EP_endRing Emergency planning end ring
 * @param d_ground_deposit Ground deposition data
 * @param dEP Evacuation parameters (timing, speeds)
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

    // Check if evacuee has reached boundary
    if (p.r >= d_radius[EP_endRing - 1]) {
        p.speed = 0.0f;
        return;
    }

    // Find current radial ring
    int rad_idx = 0;
    for (int i = 0; i < numRad; ++i) {
        if (p.r <= d_radius[i]) {
            rad_idx = i;
            break;
        }
    }

    if (p.rad0 >= dEP->evaEndRing) return;

    // Determine evacuation phase based on time

    // Phase 1: Pre-shelter delay
    if (currentTime < dEP->alarmTime + dEP->shelterDelay[p.rad0]) {
        p.speed = 0.0f;
        p.flag = 0;
    }
    // Phase 2: Sheltering
    else if (currentTime < dEP->alarmTime + dEP->shelterDelay[p.rad0] + dEP->shelterDuration[p.rad0]) {
        p.speed = 0.0f;
        p.flag = 1;
    }
    // Phase 3: Evacuation with speed profile
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

    // Calculate angular sector
    int theta_idx = static_cast<int>(p.theta / (2 * PI / numTheta)) % numTheta;

    // Get evacuation direction
    int dir = d_dir[rad_idx * numTheta + theta_idx];

    // Update position based on direction
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
// CUDA Kernels - 2D Evacuation
// ====================================================================================

/**
 * 2D evacuation calculation kernel
 * Handles multiple simulations with evacuees
 *
 * Thread organization:
 *   - blockIdx.x: Simulation index
 *   - threadIdx.x + blockIdx.y * blockDim.x: Evacuee index within simulation
 *
 * @param d_puffs_RCAP Puff data (unused)
 * @param d_dir Direction array
 * @param d_evacuee Evacuee data
 * @param d_radius Radial distances
 * @param numRad Number of radial rings
 * @param numTheta Number of angular sectors
 * @param dnop Total number of evacuees
 * @param evaEndRing Evacuation end ring
 * @param EP_endRing Emergency planning end ring
 * @param d_ground_deposit Ground deposition data
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

#endif // GPUFF_KERNELS_EVACUATION_CUH