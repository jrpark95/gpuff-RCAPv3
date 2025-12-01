// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel - Atmospheric Transport
// ====================================================================================
//
// File: gpuff_kernels_transport.cuh
// Purpose: GPU kernels for puff transport and wind field interactions
//
// This file contains:
//   - Puff flag updates and activation
//   - Wind field interpolation and advection
//   - RCAP-specific transport kernels
//   - Polar coordinate transport
//   - Validation kernels for testing
//
// ====================================================================================

#ifndef GPUFF_KERNELS_TRANSPORT_CUH
#define GPUFF_KERNELS_TRANSPORT_CUH

#include "gpuff_kernels_constants.cuh"

// ====================================================================================
// CUDA Kernels - Puff Activation
// ====================================================================================

/**
 * Update puff activation flags based on simulation progress
 * Puffs are activated gradually as simulation time progresses
 *
 * Thread organization: 1D grid, one thread per puff
 *
 * @param d_puffs Array of puff center data
 * @param activationRatio Ratio of puffs to activate (0 to 1)
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
    if (p.flag == 1) return;

    if (p.releasetime < currentTime) {
        p.flag = 1;
    }
}

// ====================================================================================
// CUDA Kernels - Standard Puff Transport
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

/**
 * Simplified validation kernel for puff transport
 * Uses constant wind field for testing
 *
 * @param d_puffs Array of puff center data
 */
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

// ====================================================================================
// CUDA Kernels - RCAP Transport
// ====================================================================================

/**
 * Move puffs using polar coordinate wind fields for RCAP
 * Simple version using pre-stored wind direction and velocity
 *
 * @param d_puffs Array of puff center data
 * @param d_RCAP_windir Wind direction array (radians)
 * @param d_RCAP_winvel Wind velocity array (m/s)
 * @param d_radi Radial distance array
 */
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

/**
 * Advanced RCAP transport kernel with deposition and decay
 * Includes dispersion parameter updates and boundary checks
 *
 * Thread organization: 2D grid (simulations x puffs per simulation)
 * Block.x = simulation index, ThreadIdx.x = puff index within simulation
 *
 * @param d_puffs_RCAP Array of RCAP puff structures
 * @param d_Vdepo Deposition velocities by particle size
 * @param d_particleSizeDistr Particle size distribution by element
 * @param EP_endRing End ring for emergency planning zone
 * @param d_ground_deposit Ground deposition array
 * @param d_ND Nuclide data array
 * @param d_radius Radial distances
 * @param numRad Number of radial rings
 * @param numTheta Number of angular sectors
 */
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

    // Check boundary conditions (5km radius)
    if (p.x * p.x + p.y * p.y > 5000.0 * 5000.0) {
        p.flag = 0;
        return;
    }

    // Update virtual distance and dispersion parameters
    p.virtual_distance += p.windvel * d_dt;

    if (d_isPG) {
        p.sigma_h = Sigma_h_Pasquill_Gifford(p.stab - 1, p.virtual_distance);
        p.sigma_z = Sigma_z_Pasquill_Gifford(p.stab - 1, p.virtual_distance);
    }

    // Wet scavenging factor
    float wetf = expf(-d_wc1*powf(p.rain, d_wc2)*d_dt);

    // Calculate grid indices for deposition
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

    // Process each nuclide
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

            // Calculate size-dependent deposition
            for (int size_idx = 0; size_idx < PARTICLE_COUNT; ++size_idx) {
                float fraction = d_particleSizeDistr[p.unitidx * (ELEMENT_COUNT * PARTICLE_COUNT)
                                                    + (group - 1) * PARTICLE_COUNT + size_idx];
                float vdep = d_Vdepo[size_idx];
                float f_size = expf(-vdep * d_dt / 1500.0f);
                f_total += fraction * f_size;
            }

            // Apply wet and dry deposition
            float new_conc = nuclide.wet_deposition ? conc * f_total * wetf : conc * f_total;
            p.conc[nuc_idx] = new_conc * decay_factor;

            // Track deposition (commented out in original)
            // float deposition = conc - new_conc;
            // atomicAdd(&d_ground_deposit[...], deposition);
        }
    }
}

/**
 * Variable wind field transport kernel with time-dependent winds
 * Uses meteorological data with temporal variation
 *
 * @param d_puffs Array of puff center data
 * @param device_meteorological_data_pres Pressure-level met data
 * @param device_meteorological_data_unis Surface-level met data
 * @param device_meteorological_data_etas Eta-coordinate met data
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

    // Boundary checks
    if (xidx < 2 || xidx > dimX-2) {
        printf("xidx error! xidx = %d\n", xidx);
    }

    if (yidx < 2 || yidx > dimY - 2) {
        printf("yidx error! yidx = %d\n", yidx);
    }

    // Find vertical levels
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

    // Bilinear interpolation weights
    float x0 = p.x / 1500.0 - xidx;
    float y0 = p.y / 1500.0 - yidx;
    float x1 = 1 - x0;
    float y1 = 1 - y0;

    // Interpolate wind components
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

    // Update position
    p.x += xwind * d_dt;
    p.y += ywind * d_dt;
    p.z += zwind * d_dt;

    // Enforce minimum height
    if (p.z < 2.0) p.z = 2.0;
}

#endif // GPUFF_KERNELS_TRANSPORT_CUH