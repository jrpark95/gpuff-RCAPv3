// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel - Deposition and Decay
// ====================================================================================
//
// File: gpuff_kernels_deposition.cuh
// Purpose: GPU kernels for dry/wet deposition and radioactive decay
//
// This file contains:
//   - Dry deposition based on gravitational settling
//   - Wet scavenging (washout) by precipitation
//   - Radioactive decay calculations
//   - Ground deposition tracking
//   - Validation kernels for testing
//
// ====================================================================================

#ifndef GPUFF_KERNELS_DEPOSITION_CUH
#define GPUFF_KERNELS_DEPOSITION_CUH

#include "gpuff_kernels_constants.cuh"

// ====================================================================================
// CUDA Kernels - Dry Deposition
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
 * Simplified validation kernel for dry deposition
 * Uses fixed mixing height for testing
 *
 * @param d_puffs Array of puff center data
 */
__global__ void dry_deposition_val(Gpuff::Puffcenter* d_puffs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    // Fixed mixing height of 1000m for validation
    p.conc *= exp(-p.drydep_vel*d_dt/1000.0);
}

// ====================================================================================
// CUDA Kernels - Wet Scavenging
// ====================================================================================

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
 * Simplified validation kernel for wet scavenging
 * Uses fixed washout rate for testing
 *
 * @param d_puffs Array of puff center data
 */
__global__ void wet_scavenging_val(Gpuff::Puffcenter* d_puffs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    // Fixed washout coefficient for validation
    float Lambda = 3.5e-5*(1.0-0.8)/(1.0-0.8);
    p.conc *= exp(-Lambda*d_dt);
}

// ====================================================================================
// CUDA Kernels - Radioactive Decay
// ====================================================================================

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

/**
 * Simplified validation kernel for radioactive decay
 * Uses decay constant stored in puff
 *
 * @param d_puffs Array of puff center data
 */
__global__ void radioactive_decay_val(Gpuff::Puffcenter* d_puffs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    p.conc *= exp(-p.decay_const*d_dt);
}

// ====================================================================================
// CUDA Kernels - Ground Deposition
// ====================================================================================

/**
 * Calculate and apply ground deposition for multiple nuclides
 * Updates ground deposit array and applies decay to deposited material
 *
 * Thread organization: 2D grid (theta x radial distance)
 * Memory access: Atomic operations for ground deposition accumulation
 *
 * @param ground_deposit Ground deposition array [nuclide][theta][radial]
 * @param d_ND Nuclide data array containing decay constants
 * @param numTheta Number of angular sectors
 * @param numRad Number of radial rings
 */
__global__ void decayGroundDeposit(float* ground_deposit, NuclideData* d_ND, int numTheta, int numRad) {
    int theta_idx = blockIdx.x;
    int rad_idx = threadIdx.x;

    if (theta_idx >= numTheta || rad_idx >= numRad) return;

    // Process each nuclide
    for (int nuc_idx = 0; nuc_idx < MAX_NUCLIDES; ++nuc_idx) {
        int deposit_idx = theta_idx * numRad * MAX_NUCLIDES +
                         rad_idx * MAX_NUCLIDES + nuc_idx;

        float current_deposit = ground_deposit[deposit_idx];
        if (current_deposit > 0.0f) {
            // Apply radioactive decay to ground deposit
            float half_life = d_ND[nuc_idx].half_life;
            if (half_life > 0.0f) {
                float decay_factor = expf(-logf(2.0f) / half_life * d_dt);
                ground_deposit[deposit_idx] *= decay_factor;
            }
        }
    }
}

/**
 * Track puff entry/exit times through radial rings for RCAP
 * Calculates deposition fractions as puffs cross ring boundaries
 *
 * Thread organization: 1D grid, one thread per puff
 * Memory access: Sequential access to puff deposition arrays
 *
 * @param d_puffs Array of puff center data
 * @param d_RCAP_windir Wind direction array
 * @param d_RCAP_winvel Wind velocity array
 * @param d_radi Radial distance array
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

    // Track entry time for first ring
    if(p.tin[0]<1.0e-8) {
        p.tin[0] = currentTime;
    }

    // Calculate distances to ring boundaries
    float hdist = 1.0*d_radi[p.head_radidx] + 0.0*d_radi[p.head_radidx+1]
                   - sqrt(p.x*p.x+p.y*p.y) - p.sigma_h*2.15;
    float tdist = - sqrt(p.x*p.x+p.y*p.y) + 1.0*d_radi[p.tail_radidx]
                    + 0.0*d_radi[p.tail_radidx+1] + p.sigma_h*2.15;

    // Check if puff head crosses ring boundary
    if (hdist * p.head_dist < 0) {
        p.tin[p.head_radidx+1] = currentTime;
        p.head_radidx++;
        hdist = 0.0;
    }

    // Check if puff tail crosses ring boundary
    if (tdist * p.tail_dist < 0) {
        p.tout[p.tail_radidx] = currentTime;
        tdist = 0.0;

        // Calculate deposition fractions
        float H = 1000.0;  // Mixing height
        float rain = 1.0;  // Rain rate
        float C1 = 1.89e-5;
        float C2 = 0.664;

        for (int iNuclide = 0; iNuclide < 9; iNuclide++) {
            if (iNuclide == 0) continue;  // Skip first nuclide

            // Wet deposition fraction
            p.fw[iNuclide][p.tail_radidx] = exp(-C1 * pow(rain, C2) *
                                                (p.tout[p.tail_radidx] - p.tin[p.tail_radidx]));

            // Dry deposition fraction for each particle size
            for (int iSize = 0; iSize < 10; iSize++) {
                p.fd[iNuclide][p.tail_radidx] = exp(-d_vdepo[iSize] *
                                                    (p.tout[p.tail_radidx] - p.tin[p.tail_radidx]) / H);
                p.fallout[iNuclide][p.tail_radidx] += d_size[iNuclide][iSize] *
                                                      (1 - p.fd[iNuclide][p.tail_radidx] * p.fw[iNuclide][p.tail_radidx]);
            }

            // Update concentration after deposition
            p.conc_arr[iNuclide] = p.conc_arr[iNuclide] * (1 - p.fallout[iNuclide][p.tail_radidx]);
        }
        p.tail_radidx++;
    }

    // Update position for next timestep
    float xwind = p.windvel*cos(p.windir);
    float ywind = p.windvel*sin(p.windir);

    p.x += xwind*d_dt;
    p.y += ywind*d_dt;

    p.head_dist = hdist;
    p.tail_dist = tdist;
}

#endif // GPUFF_KERNELS_DEPOSITION_CUH