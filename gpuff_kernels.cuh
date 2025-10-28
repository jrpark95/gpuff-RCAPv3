// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel Implementation
// ====================================================================================
//
// File: gpuff_kernels.cuh
// Purpose: GPU kernel implementations for Gaussian puff atmospheric dispersion model
//          with radiological consequence assessment capabilities
//
// This file contains CUDA kernels for:
//   - Atmospheric dispersion calculations (Pasquill-Gifford and Briggs-McElroy-Pooler)
//   - Puff transport by wind field interpolation
//   - Radioactive decay and deposition (wet/dry)
//   - Concentration accumulation on receptor grids
//   - Evacuee dose calculations for emergency planning
//   - Nuclide-specific exposure computations
//
// Physics Models Implemented:
//   - Gaussian puff dispersion model
//   - Pasquill-Gifford stability classes (A-G)
//   - Atmospheric stability classification via temperature gradient
//   - Bilinear interpolation for meteorological fields
//   - Radioactive decay chain calculations
//   - Ground deposition modeling
//
// Performance Notes:
//   - Kernels optimized for CUDA memory coalescing
//   - Shared memory used for parallel reductions
//   - Atomic operations for grid accumulation
//   - Thread indexing patterns preserved for GPU efficiency
//
// ====================================================================================

#include "gpuff.cuh"
#include "gpuff_kernels.h"

// ====================================================================================
// Constants
// ====================================================================================

// Grid spacing for meteorological data (meters)
constexpr float GRID_SPACING = 1500.0f;

// Minimum puff height (meters)
constexpr float MIN_PUFF_HEIGHT = 2.0f;

// Convergence tolerance for Newton-Raphson iteration
constexpr float NEWTON_RAPHSON_TOLERANCE = 1e-4f;

// Pasquill-Gifford stability classes
constexpr int STABILITY_CLASS_A = 0;  // Extremely unstable
constexpr int STABILITY_CLASS_B = 1;  // Moderately unstable
constexpr int STABILITY_CLASS_C = 2;  // Slightly unstable
constexpr int STABILITY_CLASS_D = 3;  // Neutral
constexpr int STABILITY_CLASS_E = 4;  // Slightly stable
constexpr int STABILITY_CLASS_F = 5;  // Moderately stable
constexpr int STABILITY_CLASS_G = 6;  // Extremely stable

// Wet scavenging parameters
constexpr float WET_SCAVENGING_LAMBDA_COEFF = 3.5e-5f;
constexpr float WET_SCAVENGING_RH_THRESHOLD = 0.8f;

// ====================================================================================
// Device Helper Functions - Atomic Operations
// ====================================================================================

/**
 * Atomic minimum operation for floating-point values
 * Uses Compare-And-Swap (CAS) to atomically update minimum value
 *
 * Thread-safe operation for finding minimum across GPU threads
 */
__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;

    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(val));
    }

    return __int_as_float(old);
}

/**
 * Atomic maximum operation for floating-point values
 * Uses Compare-And-Swap (CAS) to atomically update maximum value
 *
 * Thread-safe operation for finding maximum across GPU threads
 */
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;

    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(val));
    }

    return __int_as_float(old);
}

// ====================================================================================
// Device Helper Functions - Pasquill-Gifford Dispersion Formulas
// ====================================================================================

/**
 * Calculate horizontal dispersion coefficient (sigma_h) using Pasquill-Gifford formula
 *
 * Formula: sigma_h = exp(c0 + c1*ln(x) + c2*ln(x)^2)
 * where x is the virtual distance from source
 *
 * Reference: Pasquill, F. (1961). "The estimation of the dispersion of windborne material"
 *            Meteorological Magazine, 90, 33-49.
 *
 * @param PasquillCategory Atmospheric stability class (0=A extremely unstable, 6=G extremely stable)
 * @param virtual_distance Effective distance traveled by puff (meters)
 * @return Horizontal dispersion parameter sigma_h (meters)
 */
__device__ float Sigma_h_Pasquill_Gifford(int PasquillCategory, float virtual_distance) {
    // Pasquill-Gifford coefficients for horizontal dispersion (A through G stability classes)
    float coefficient0[7] = {-1.104, -1.634, -2.054, -2.555, -2.754, -3.143, -3.143};
    float coefficient1[7] = {0.9878, 1.0350, 1.0231, 1.0423, 1.0106, 1.0418, 1.0418};
    float coefficient2[7] = {-0.0076, -0.0096, -0.0076, -0.0087, -0.0064, -0.0070, -0.0070};

    float log_distance = log(virtual_distance);
    float sigma = exp(coefficient0[PasquillCategory] +
                      coefficient1[PasquillCategory] * log_distance +
                      coefficient2[PasquillCategory] * log_distance * log_distance);

    return sigma;
}

/**
 * CPU version of horizontal dispersion coefficient calculation
 * Identical formula to device version, for validation and debugging
 */
float Sigma_h_Pasquill_Gifford_cpu(int PasquillCategory, float virtual_distance) {
    float coefficient0[7] = {-1.104, -1.634, -2.054, -2.555, -2.754, -3.143, -3.143};
    float coefficient1[7] = {0.9878, 1.0350, 1.0231, 1.0423, 1.0106, 1.0418, 1.0418};
    float coefficient2[7] = {-0.0076, -0.0096, -0.0076, -0.0087, -0.0064, -0.0070, -0.0070};

    float log_distance = log(virtual_distance);
    float sigma = exp(coefficient0[PasquillCategory] +
                      coefficient1[PasquillCategory] * log_distance +
                      coefficient2[PasquillCategory] * log_distance * log_distance);

    return sigma;
}

/**
 * Derivative of horizontal dispersion coefficient with respect to virtual distance
 * Used in Newton-Raphson iteration for virtual distance calculation
 *
 * Formula: d(sigma_h)/dx where sigma_h = exp(c0 + c1*ln(x) + c2*ln(x)^2)
 *
 * @param PasquillCategory Atmospheric stability class (0-6)
 * @param virtual_distance Effective distance traveled by puff (meters)
 * @return Derivative d(sigma_h)/dx
 */
__device__ float dSh_PG(int PasquillCategory, float virtual_distance) {
    float coefficient0[7] = {-1.104, -1.634, -2.054, -2.555, -2.754, -3.143, -3.143};
    float coefficient1[7] = {0.9878, 1.0350, 1.0231, 1.0423, 1.0106, 1.0418, 1.0418};
    float coefficient2[7] = {-0.0076, -0.0096, -0.0076, -0.0087, -0.0064, -0.0070, -0.0070};

    float log_distance = log(virtual_distance);
    float sigma = pow(virtual_distance, coefficient1[PasquillCategory] - 1)
                  * exp(coefficient0[PasquillCategory] + coefficient2[PasquillCategory] * log_distance * log_distance)
                  * (coefficient1[PasquillCategory] + 2 * coefficient2[PasquillCategory] * log_distance);

    return sigma;
}

/**
 * Calculate vertical dispersion coefficient (sigma_z) using Pasquill-Gifford formula
 *
 * Formula: sigma_z = exp(c0 + c1*ln(x) + c2*ln(x)^2)
 * where x is the virtual distance from source
 *
 * Vertical dispersion varies significantly with stability class:
 *   - Class A (unstable): Large vertical mixing
 *   - Class G (stable): Limited vertical mixing
 *
 * @param PasquillCategory Atmospheric stability class (0-6)
 * @param virtual_distance Effective distance traveled by puff (meters)
 * @return Vertical dispersion parameter sigma_z (meters)
 */
__device__ float Sigma_z_Pasquill_Gifford(int PasquillCategory, float virtual_distance) {
    // Pasquill-Gifford coefficients for vertical dispersion (A through G stability classes)
    float coefficient0[7] = {4.679, -1.999, -2.341, -3.186, -3.783, -4.490, -4.490};
    float coefficient1[7] = {-1.172, 0.8752, 0.9477, 1.1737, 1.3010, 1.4024, 1.4024};
    float coefficient2[7] = {0.2770, 0.0136, -0.0020, -0.0316, -0.0450, -0.0540, -0.0540};

    float log_distance = log(virtual_distance);
    float sigma = exp(coefficient0[PasquillCategory] +
                      coefficient1[PasquillCategory] * log_distance +
                      coefficient2[PasquillCategory] * log_distance * log_distance);

    return sigma;
}

/**
 * CPU version of vertical dispersion coefficient calculation
 * Identical formula to device version, for validation and debugging
 */
float Sigma_z_Pasquill_Gifford_cpu(int PasquillCategory, float virtual_distance) {
    float coefficient0[7] = {4.679, -1.999, -2.341, -3.186, -3.783, -4.490, -4.490};
    float coefficient1[7] = {-1.172, 0.8752, 0.9477, 1.1737, 1.3010, 1.4024, 1.4024};
    float coefficient2[7] = {0.2770, 0.0136, -0.0020, -0.0316, -0.0450, -0.0540, -0.0540};

    float log_distance = log(virtual_distance);
    float sigma = exp(coefficient0[PasquillCategory] +
                      coefficient1[PasquillCategory] * log_distance +
                      coefficient2[PasquillCategory] * log_distance * log_distance);

    return sigma;
}

/**
 * Derivative of vertical dispersion coefficient with respect to virtual distance
 * Used in Newton-Raphson iteration for virtual distance calculation
 *
 * @param PasquillCategory Atmospheric stability class (0-6)
 * @param virtual_distance Effective distance traveled by puff (meters)
 * @return Derivative d(sigma_z)/dx
 */
__device__ float dSz_PG(int PasquillCategory, float virtual_distance) {
    float coefficient0[7] = {4.679, -1.999, -2.341, -3.186, -3.783, -4.490, -4.490};
    float coefficient1[7] = {-1.172, 0.8752, 0.9477, 1.1737, 1.3010, 1.4024, 1.4024};
    float coefficient2[7] = {0.2770, 0.0136, -0.0020, -0.0316, -0.0450, -0.0540, -0.0540};

    float log_distance = log(virtual_distance);
    float sigma = pow(virtual_distance, coefficient1[PasquillCategory] - 1)
                  * exp(coefficient0[PasquillCategory] + coefficient2[PasquillCategory] * log_distance * log_distance)
                  * (coefficient1[PasquillCategory] + 2 * coefficient2[PasquillCategory] * log_distance);

    return sigma;
}

// ====================================================================================
// Device Helper Functions - Briggs-McElroy-Pooler Dispersion Formulas
// ====================================================================================

/**
 * Calculate vertical dispersion coefficient (sigma_z) using Briggs-McElroy-Pooler formula
 *
 * Alternative to Pasquill-Gifford, accounts for urban vs rural terrain effects
 * Formula: sigma_z = c0 * x * (1 + c1*x)^c2
 *
 * Reference: Briggs, G.A. (1973). "Diffusion estimation for small emissions"
 *            ATDL Contribution File No. 79, Air Resources Atmospheric Turbulence
 *            and Diffusion Laboratory, Oak Ridge, TN.
 *
 * @param PasquillCategory Atmospheric stability class (0-6)
 * @param virtual_distance Effective distance traveled by puff (meters)
 * @return Vertical dispersion parameter sigma_z (meters)
 */
__device__ float Sigma_z_Briggs_McElroy_Pooler(int PasquillCategory, float virtual_distance) {
    // Coefficients for rural terrain
    float coefficient0_rural[7] = {0.20, 0.12, 0.08, 0.06, 0.03, 0.016, 0.016};
    float coefficient1_rural[7] = {0.0, 0.0, 0.0002, 0.0015, 0.0003, 0.0003, 0.0003};
    float coefficient2_rural[7] = {1.0, 1.0, -0.5, -0.5, -1.0, -1.0, -1.0};

    // Coefficients for urban terrain
    float coefficient0_urban[7] = {0.24, 0.24, 0.2, 0.14, 0.08, 0.08, 0.08};
    float coefficient1_urban[7] = {0.001, 0.001, 0.0, 0.0003, 0.00015, 0.00015, 0.00015};
    float coefficient2_urban[7] = {0.5, 0.5, 1.0, -0.5, -0.5, -0.5, -0.5};

    float sigma;

    if (d_isRural) {
        sigma = coefficient0_rural[PasquillCategory] * virtual_distance *
                pow(1 + coefficient1_rural[PasquillCategory] * virtual_distance, coefficient2_rural[PasquillCategory]);
    }
    else {
        sigma = coefficient0_urban[PasquillCategory] * virtual_distance *
                pow(1 + coefficient1_urban[PasquillCategory] * virtual_distance, coefficient2_urban[PasquillCategory]);
    }

    return sigma;
}

/**
 * Derivative of Briggs-McElroy-Pooler vertical dispersion with respect to virtual distance
 * Used in Newton-Raphson iteration
 *
 * @param PasquillCategory Atmospheric stability class (0-6)
 * @param virtual_distance Effective distance traveled by puff (meters)
 * @return Derivative d(sigma_z)/dx for Briggs-McElroy-Pooler formula
 */
__device__ float dSz_BMP(int PasquillCategory, float virtual_distance) {
    float coefficient0_rural[7] = {0.20, 0.12, 0.08, 0.06, 0.03, 0.016, 0.016};
    float coefficient1_rural[7] = {0.0, 0.0, 0.0002, 0.0015, 0.0003, 0.0003, 0.0003};
    float coefficient2_rural[7] = {1.0, 1.0, -0.5, -0.5, -1.0, -1.0, -1.0};

    float coefficient0_urban[7] = {0.24, 0.24, 0.2, 0.14, 0.08, 0.08, 0.08};
    float coefficient1_urban[7] = {0.001, 0.001, 0.0, 0.0003, 0.00015, 0.00015, 0.00015};
    float coefficient2_urban[7] = {0.5, 0.5, 1.0, -0.5, -0.5, -0.5, -0.5};

    float sigma;

    if (d_isRural) {
        sigma = pow(coefficient1_rural[PasquillCategory] * virtual_distance + 1, coefficient2_rural[PasquillCategory] - 1) *
                (coefficient0_rural[PasquillCategory] * coefficient1_rural[PasquillCategory] *
                 (coefficient2_rural[PasquillCategory] + 1) * virtual_distance + coefficient0_rural[PasquillCategory]);
    }
    else {
        sigma = pow(coefficient1_urban[PasquillCategory] * virtual_distance + 1, coefficient2_urban[PasquillCategory] - 1) *
                (coefficient0_urban[PasquillCategory] * coefficient1_urban[PasquillCategory] *
                 (coefficient2_urban[PasquillCategory] + 1) * virtual_distance + coefficient0_urban[PasquillCategory]);
    }

    return sigma;
}

// ====================================================================================
// Device Helper Functions - Newton-Raphson Iteration
// ====================================================================================

/**
 * Newton-Raphson iteration to find virtual distance from target horizontal dispersion
 *
 * Solves: sigma_h(x) = target_sigma for x (virtual distance)
 * Uses iterative method: x_new = x - f(x)/f'(x)
 * where f(x) = sigma_h(x) - target_sigma
 *
 * This is needed because puffs track sigma values but need to compute
 * equivalent virtual distances for dispersion updates
 *
 * @param PasquillCategory Atmospheric stability class (0-6)
 * @param targetSigma Desired horizontal dispersion parameter (meters)
 * @param init Initial guess for virtual distance (meters)
 * @return Calculated virtual distance (meters)
 */
__device__ float NewtonRaphson_h(int PasquillCategory, float targetSigma, float init) {
    float x = init;
    float fx, dfx;

    while (true) {
        if (d_isPG) {
            fx = Sigma_h_Pasquill_Gifford(PasquillCategory, x) - targetSigma;
            dfx = dSh_PG(PasquillCategory, x);
        }

        x = x - fx / dfx;

        if (fabs(fx) < NEWTON_RAPHSON_TOLERANCE) {
            break;
        }
    }

    return x;
}

/**
 * Newton-Raphson iteration to find virtual distance from target vertical dispersion
 *
 * Solves: sigma_z(x) = target_sigma for x (virtual distance)
 *
 * @param PasquillCategory Atmospheric stability class (0-6)
 * @param targetSigma Desired vertical dispersion parameter (meters)
 * @param init Initial guess for virtual distance (meters)
 * @return Calculated virtual distance (meters)
 */
__device__ float NewtonRaphson_z(int PasquillCategory, float targetSigma, float init) {
    float x = init;
    float fx, dfx;

    while (true) {
        if (d_isPG) {
            fx = Sigma_z_Pasquill_Gifford(PasquillCategory, x) - targetSigma;
            dfx = dSz_PG(PasquillCategory, x);
        }
        else {
            fx = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, x) - targetSigma;
            dfx = dSz_BMP(PasquillCategory, x);
        }

        x = x - fx / dfx;

        if (fabs(fx) < NEWTON_RAPHSON_TOLERANCE) {
            break;
        }
    }

    return x;
}

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
    if (p.flag == 1) return;

    if (p.releasetime < currentTime) {
        p.flag = 1;
    }
}

/**
 * CPU version of puff flag update for RCAP simulations
 * Identical logic to device version, for validation
 */
void Gpuff::update_puff_flags2_cpu(float currentTime, int nop) {
    for (int idx = 0; idx < nop; ++idx) {
        Gpuff::Puffcenter_RCAP& p = puffs_RCAP[idx];
        if (p.flag == 1) continue;

        if (p.releasetime < currentTime) {
            p.flag = 1;
        }
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

// __global__ void accumulateConc(
//     Gpuff::Puffcenter* puffs, 
//     RectangleGrid::GridPoint* d_grid, 
//     float* concs, 
//     int ngrid)
// {
//     int puffIdx = blockIdx.y * blockDim.y + threadIdx.y;
//     int gridIdx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (puffIdx >= d_nop || gridIdx >= ngrid) return;

//     Gpuff::Puffcenter& p = puffs[puffIdx];
//     RectangleGrid::GridPoint& g = d_grid[gridIdx];

//     printf("%f, %f\n", p.x, g.y);

//     if (puff.flag){
//         float dx = g.x - p.x;
//         float dy = g.y - p.y;
//         float distSq = dx * dx + dy * dy;

//         if (distSq != 0.0f){
//             float contribution = 1.0f / distSq;
//             atomicAdd(&concs[gridIdx], contribution);
//         }
//     }
// }

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

    //int tt = floor((float)p.timeidx/(float)d_nop*(float)d_time_end/3600.0);

    // float xwind = d_RCAP_winvel[timeidx]*cos(d_RCAP_windir[timeidx]);
    // float ywind = d_RCAP_winvel[timeidx]*sin(d_RCAP_windir[timeidx]);

    // float xwind = d_RCAP_winvel[tt]*cos(d_RCAP_windir[tt]);
    // float ywind = d_RCAP_winvel[tt]*sin(d_RCAP_windir[tt]);

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

    if (p.x * p.x + p.y * p.y > 5000.0 * 5000.0) {
        p.flag = 0;
        return;
    }

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

    //printf("p.rain = %f, wetf = %f\n", p.rain, wetf);

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
        //float decay_factor = 1.0;


        int group = nuclide.chemical_group;
        if (group < 1 || group > ELEMENT_COUNT) {
            continue;
        }

        if (nuclide.dry_deposition == true) {
            float conc = p.conc[nuc_idx];

            //printf("d_nop = %d, idx = %d, conc[%d] = %e\n", d_nop, idx, nuc_idx, conc);

            float f_total = 0.0f;

            for (int size_idx = 0; size_idx < PARTICLE_COUNT; ++size_idx) {

                float fraction = d_particleSizeDistr[p.unitidx * (ELEMENT_COUNT * PARTICLE_COUNT) 
                                                    + (group - 1) * PARTICLE_COUNT + size_idx];
                float vdep = d_Vdepo[size_idx];

                float f_size = expf(-vdep * d_dt / 1500.0f);

                //printf("idx = %d, vdep = %e, conc = %e, fraction = %e\n", idx, vdep, p.conc[nuc_idx], fraction);
                //printf("idx = %d, nucidx = %d, f_size = %e, f_total = %e\n", idx, nuc_idx, f_size, f_total);

                f_total += fraction * f_size;
            }

            //if (nuclide.wet_deposition == true) p.conc[nuc_idx] = conc * f_total * wetf;
            //else p.conc[nuc_idx] = conc * f_total;

            float new_conc = nuclide.wet_deposition ? conc * f_total * wetf : conc * f_total;
            //float new_conc = conc * f_total;
            p.conc[nuc_idx] = new_conc * decay_factor;

            float deposition = conc - new_conc;

            //atomicAdd(&d_ground_deposit[ simidx * numRad * numTheta * MAX_NUCLIDES + 
            //    rad_idx * numTheta * MAX_NUCLIDES + 
            //    theta_idx * MAX_NUCLIDES + nuc_idx], deposition);

        }

        // if(idx==0 && nuc_idx==0) printf("%e\n", p.conc[nuc_idx]);
        //if (idx == 0 && nuc_idx == 0) printf("%e\n", d_ground_deposit[ 1 * simidx * numRad * numTheta * MAX_NUCLIDES
        //    + 1 * numTheta * MAX_NUCLIDES + 8 * MAX_NUCLIDES + nuc_idx]);


        //printf("rad = %d, theta = %d\n", rad_idx, theta_idx)

    }

}

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
            //printf("p.sigma_h = %e\n", p.sigma_h);
            p.sigma_z = Sigma_z_Pasquill_Gifford_cpu(p.stab - 1, p.virtual_distance);
            //printf("p.sigma_z = %e\n", p.sigma_z);

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
                //int deposit_index = idx * numRad * numTheta * MAX_NUCLIDES +
                //    rad_idx * numTheta * MAX_NUCLIDES +
                //    theta_idx * MAX_NUCLIDES + nuc_idx;

                //ground_deposit[deposit_index] += deposition;
            }
        }
    }
}


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

    //if (p.flag == 0) {
    //    return;
    //}

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

    //printf("time = %f, 1 = %f, 2 = %f\n", currentTime, dEP->alarmTime + dEP->shelterDelay[rad_idx], dEP->alarmTime + dEP->shelterDelay[rad_idx] + dEP->shelterDuration[rad_idx]);
        //printf("speed = %f, del = %f, dur = %f\n", p.speed, dEP->shelterDelay[rad_idx], dEP->shelterDuration[rad_idx]);
    

    if (currentTime < dEP->alarmTime + dEP->shelterDelay[p.rad0]) {
        p.speed = 0.0f;
        p.flag = 0;
    }
    else if (currentTime < dEP->alarmTime + dEP->shelterDelay[p.rad0] + dEP->shelterDuration[p.rad0]) {
        p.speed = 0.0f;
        p.flag = 1;

    }
    //else {
    //    p.speed = dEP->speeds[0];
    //    p.flag = 2;
    //}
    else {
        for (int i = 0; i < dEP->nSpeedPeriod-1; i++) {
            // printf("i = %d\n", i);
            float sum_durations = 0.0f;
            for (int j = 0; j < i+1; j++) sum_durations += dEP->durations[j];
            if (currentTime < dEP->alarmTime + dEP->shelterDelay[p.rad0] + dEP->shelterDuration[p.rad0]
                + sum_durations) {
                p.speed = dEP->speeds[i];
                p.flag = 2;
                //printf("time = %f, du = %f, speed = %f, i = %d\n", currentTime, dEP->alarmTime + dEP->shelterDelay[p.rad0] + dEP->shelterDuration[p.rad0]
                //    + sum_durations, p.speed, i);
                break;
            }
        }
        float sum_durations = 0.0f;
        for (int j = 0; j < dEP->nSpeedPeriod-1; j++) sum_durations += dEP->durations[j];
        if (currentTime > dEP->alarmTime + dEP->shelterDelay[p.rad0] + dEP->shelterDuration[p.rad0]
            + sum_durations) {
            p.speed = dEP->speeds[dEP->nSpeedPeriod-1];
            p.flag = 2;
            //printf("time = %f, du = %f, speed = %f, i = %d\n", currentTime, dEP->alarmTime + dEP->shelterDelay[p.rad0] + dEP->shelterDuration[p.rad0]
            //    + sum_durations, p.speed, dEP->nSpeedPeriod-1);
        }
    }



    int theta_idx = static_cast<int>(p.theta / (2 * PI / numTheta)) % numTheta;

    int dir = d_dir[rad_idx * numTheta + theta_idx];

    //printf("dir[%d][%d] = %d\n", rad_idx, theta_idx, dir);


    switch (dir) {
    case DIR_F: // Forward
        p.r += p.speed * d_dt;
        //printf("Evacuee %d moving forward to r = %f\n", idx, p.r);
        //printf("p.speed = %f\n", p.speed);
        break;
    case DIR_B: // Backward
        p.r -= p.speed * d_dt;
        //printf("Evacuee %d moving backward to r = %f\n", idx, p.r);
        //printf("p.speed = %f\n", p.speed);
        if (p.r < 0) p.r = 0;
        break;
    case DIR_L: // Left
        p.theta -= p.speed * d_dt / p.r;
        //printf("Evacuee %d moving left to theta = %f\n", idx, p.theta);
        //printf("p.speed = %f\n", p.speed);
        if (p.theta < 0) p.theta += 2 * PI;
        break;
    case DIR_R: // Right
        p.theta += p.speed * d_dt / p.r;
        //printf("Evacuee %d moving right to theta = %f\n", idx, p.theta);
        //printf("p.speed = %f\n", p.speed);
        if (p.theta >= 2 * PI) p.theta -= 2 * PI;
        break;
    default: // DIR_NONE or any other invalid direction
        // No movement
        break;
    }

}

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
        //int direction = ED.directions[rad_idx * numTheta + theta_idx];
        int direction = ED.get(rad_idx, theta_idx);


        //printf("direction = %d\n", ED.get(rad_idx, theta_idx));
        //printf("DIR_F = %d\n", DIR_F);

        //printf("p.speed = %f, dt = %f\n", p.speed, dt);

        switch (direction) {
        case DIR_F: // Forward
            p.r += p.speed * dt;
            //printf("DIR_F\n");

            break;
        case DIR_B: // Backward
            p.r -= p.speed * dt;
            //printf("DIR_B\n");

            if (p.r < 0) p.r = 0;
            break;
        case DIR_L: // Left
            p.theta -= p.speed * dt / p.r;
            //printf("DIR_L\n");

            if (p.theta < 0) p.theta += 2 * PI;
            break;
        case DIR_R: // Right
            p.theta += p.speed * dt / p.r;
            //printf("DIR_R\n");

            if (p.theta >= 2 * PI) p.theta -= 2 * PI;
            break;
        default: // DIR_NONE or any other invalid direction
            // No movement
            break;
        }
    }
}



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
    // float tdist = 0.5*d_radi[p.tail_radidx] + 0.5*d_radi[p.tail_radidx+1] - sqrt(p.x*p.x+p.y*p.y) + p.sigma_h*2.15;
    float tdist = - sqrt(p.x*p.x+p.y*p.y) + 1.0*d_radi[p.tail_radidx] 
                    + 0.0*d_radi[p.tail_radidx+1] + p.sigma_h*2.15;

    //if(hdist<0) {
    if (hdist * p.head_dist < 0) {

        if(idx==1) {
            printf("hdist = %f, head_dist = %f\n", hdist, p.head_dist);
            printf("head_radidx = %d, currentTime = %f\n", p.head_radidx, currentTime);
        }
        p.tin[p.head_radidx+1] = currentTime;
        p.head_radidx++;
        hdist = 0.0;
    }
    //if(tdist<0) {
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

        for (int iNuclide = 0; iNuclide < 9; iNuclide++) {
            if (iNuclide == 0) continue;
            p.fw[iNuclide][p.tail_radidx] = exp(-C1 * pow(rain, C2) * (p.tout[p.tail_radidx] - p.tin[p.tail_radidx]));

            for (int iSize = 0; iSize < 10; iSize++) {

                p.fd[iNuclide][p.tail_radidx] = exp(-d_vdepo[iSize] * (p.tout[p.tail_radidx] - p.tin[p.tail_radidx]) / H);
                p.fallout[iNuclide][p.tail_radidx] += d_size[iNuclide][iSize] * (1 - p.fd[iNuclide][p.tail_radidx] * p.fw[iNuclide][p.tail_radidx]);
                //p.fallout[p.tail_radidx] = (1 - p.fd[p.tail_radidx] * p.fw[p.tail_radidx]);

            }
            //p.conc -= p.conc*p.fallout[p.tail_radidx]; // need multiply "DCF[iNuclide]"
            p.conc_arr[iNuclide] = p.conc_arr[iNuclide] * (1 - p.fallout[iNuclide][p.tail_radidx]);
        }
        //p.conc = p.conc * (1-p.fallout[p.tail_radidx]); // need multiply "DCF[iNuclide]"
        p.tail_radidx++;
    }

    //if (idx == 10) {
    //    //printf("tdist = %f, tail_dist = %f\n", tdist, p.tail_dist);
    //    //printf("tail_radidx = %d, currentTime = %f\n", p.tail_radidx, currentTime);
    //    //printf("tout[%d] = %f\n", p.tail_radidx, p.tout[p.tail_radidx - 1]);
    //}

    float xwind = p.windvel*cos(p.windir);
    float ywind = p.windvel*sin(p.windir);


    p.x += xwind*d_dt;
    p.y += ywind*d_dt;

    p.head_dist = hdist;
    p.tail_dist = tdist;

}

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

    // int tt = floor((float)p.timeidx/(float)d_nop*(float)d_time_end/3600.0);

    // // float xwind = d_RCAP_winvel[timeidx]*cos(d_RCAP_windir[timeidx]);
    // // float ywind = d_RCAP_winvel[timeidx]*sin(d_RCAP_windir[timeidx]);

    // float xwind = d_RCAP_winvel[tt]*cos(d_RCAP_windir[tt]);
    // float ywind = d_RCAP_winvel[tt]*sin(d_RCAP_windir[tt]);

    float xwind = p.windvel*cos(p.windir);
    float ywind = p.windvel*sin(p.windir);

    float vel = sqrt(xwind*xwind + ywind*ywind);

    

    int PasquillCategory = p.stab-1;

    float new_virtual_distance_h = NewtonRaphson_h(PasquillCategory, p.sigma_h, p.virtual_distance) + vel*d_dt;
    //float new_virtual_distance_z = NewtonRaphson_z(PasquillCategory, p.sigma_z, p.virtual_distance) + vel*d_dt;

    if(d_isPG){
        p.sigma_h = Sigma_h_Pasquill_Gifford(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Pasquill_Gifford(PasquillCategory, new_virtual_distance_z);
    }
    else{
        //p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_z);
    }

    p.virtual_distance = new_virtual_distance_h;

}

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
    //float new_virtual_distance_z = NewtonRaphson_z(PasquillCategory, p.sigma_z, p.virtual_distance) + vel*d_dt;

    if (d_isPG) {
        p.sigma_h = Sigma_h_Pasquill_Gifford(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Pasquill_Gifford(PasquillCategory, new_virtual_distance_z);
    }
    else {
        //p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_z);
    }

    p.virtual_distance = new_virtual_distance_h;

}


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

    //ywind *= sqrt(sinf(currentTime / 10000));

    p.x += xwind * d_dt;
    p.y += ywind * d_dt;
    p.z += zwind * d_dt;

    if (p.z < 2.0) p.z = 2.0;
}

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

//__global__ void print_MACCS_data(const MACCSData* d_maccsData) {
//    printf("Total Nuclides: %d\n", d_maccsData->nuclide_count);
//
//    for (int i = 1; i < d_maccsData->nuclide_count; ++i) {
//        const NuclideData& nuclide = d_maccsData->nuclides[i];
//        printf("Nuclide ID: %d, Name: %s\n", nuclide.id, nuclide.name);
//        printf("  Half Life: %e seconds\n", nuclide.half_life);
//        printf("  Atomic Weight: %f g\n", nuclide.atomic_weight);
//        printf("  Chemical Group: %s\n", nuclide.chemical_group);
//        printf("  Core Inventory: %e Ci/MWth\n", nuclide.core_inventory);
//
//        // Debugging: Print organ_count
//        printf("  Organ Count: %d\n", nuclide.organ_count);
//
//        for (int k = 0; k < nuclide.organ_count; ++k) {
//            printf("    Organ Name: %s\n", nuclide.organ_names[k]);
//            printf("      Exposure Data: ");
//            for (int l = 0; l < DATA_FIELDS; ++l) {
//                printf("%e ", nuclide.exposure_data[k][l]);
//            }
//            printf("\n");
//        }
//        printf("\n");
//    }
//}

__global__ void printDeviceArray(int* d_dir, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        printf("Row %d: ", row + 1);
        for (int col = 0; col < cols; ++col) {
            printf("%d ", d_dir[row * cols + col]);
        }
        printf("\n");
    }
}

__global__ void printEvacueesKernel(const Evacuee* d_evacuees, size_t numEvacuees) {
    for (size_t idx = 0; idx < numEvacuees; ++idx) {
        printf("Evacuee %lu - Population: %f, Radius: %f, Theta: %f, Speed: %f\n",
            idx, d_evacuees[idx].population, d_evacuees[idx].r,
            d_evacuees[idx].theta, d_evacuees[idx].speed);
    }
}

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

__global__ void computeDoseForNuclide(
    float* puffs, int puffIdx, Evacuee evacuees, int evacueeIdx, float* exposure, float distance, float* sdata, int nuclideStart, int nuclideEnd) {
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

//__global__ void computeEvacueeDoseReduction3(
//    Gpuff::Puffcenter_RCAP* d_puffs_RCAP,
//    Evacuee* d_evacuees, float* d_exposure
//) {
//    extern __shared__ float sdata[];
//
//    int simIdx = blockIdx.x;
//    int evacueeIdx = blockIdx.y;
//    int puffIdx = threadIdx.x;
//
//    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim) {
//        sdata[threadIdx.x] = 0.0f;
//
//        Gpuff::Puffcenter_RCAP& puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
//        Evacuee& evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];
//
//        float dx = evacuee.r * cos(evacuee.theta) - puff.x;
//        float dy = evacuee.r * sin(evacuee.theta) - puff.y;
//        float distance = sqrt(dx * dx + dy * dy);
//
//        if (distance > 0.0f) {
//            int numNuclides = MAX_NUCLIDES;
//            int threadsPerNuclide = 256;
//            dim3 blockDims(threadsPerNuclide);
//            dim3 gridDims((numNuclides + threadsPerNuclide - 1) / threadsPerNuclide);
//
//            computeDoseForNuclide << <gridDims, blockDims, threadsPerNuclide * sizeof(float) >> > (
//                puff.conc, puffIdx, evacuee, evacueeIdx, d_exposure, distance, sdata, 0, numNuclides);
//        }
//
//        __syncthreads();
//
//        if (threadIdx.x == 0) {
//            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose = sdata[0];
//        }
//    }
//}


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
            //printf("doseSum = %f\n", doseSum);
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
        //printf("(4) = %f, (2) = %f, puff.flag = %d\n", dPF->pfactor[puff.flag][4], dPF->pfactor[puff.flag][2], puff.flag);
        //printf("sdata[%d] = %f\n", threadIdx.x, sdata[threadIdx.x]);




        __syncthreads();

/*        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }  */    


        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sdata[threadIdx.x] += __shfl_down_sync(0xffffffff, sdata[threadIdx.x], offset);
        }

        if (threadIdx.x == 0) {
            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose = sdata[0];
        }



        //// Write result for this block to global mem
        //if (threadIdx.x % warpSize == 0) { // Assuming warpSize is a divisor of blockDim.x
        //    atomicAdd(&d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose, sdata[threadIdx.x]);
        //}
    }
}

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

        //if (threadIdx.x % warpSize == 0) {
        //    atomicAdd(&d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose, val);
        //}
    }
}

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


    //printf("a = %d, b = %d, c = %d\n", d_numSims, d_totalevacuees_per_Sim, d_totalpuff_per_Sim);
    //sdata_inhalation[threadIdx.x] = 0.0f;
    //sdata_cloudshine[threadIdx.x] = 0.0f;

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

        //printf("%e\n", distanceFactor);
        float puffInhalationDose[MAX_ORGANS] = { 0.0f, };
        float puffCloudshineDose[MAX_ORGANS] = { 0.0f, };
        //if (gaussianFactor > 1e-30f) {
        if (1) {

            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffConc = puff.conc[nuclideIdx];// *gaussianFactor;

                if (puffConc > 0.0f) {
                    float totalInhalation[MAX_ORGANS] = { 0.0f, };
                    float totalCloudshine[MAX_ORGANS] = { 0.0f, };

#pragma unroll
                    for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++) {

                        float cloudshineValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 0];
                        if (cloudshineValue > 0.0f) {
                            //totalCloudshine += cloudshineValue;
                            puffCloudshineDose[organIdx] += cloudshineValue * puffConc * d_dt * distanceFactor;
                            if (dPF->pfactor[pfidx][0] > 0) puffCloudshineDose[organIdx] *= dPF->pfactor[pfidx][0];
                        }

                        float inhalationValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2];
                        if (inhalationValue > 0.0f) {
                            //totalInhalation += inhalationValue;
                            puffInhalationDose[organIdx] += inhalationValue * puffConc * d_dt * gaussianFactor;
                            if (dPF->pfactor[pfidx][4] > 0) puffInhalationDose[organIdx] *= dPF->pfactor[pfidx][4] * dPF->pfactor[pfidx][2];

                        }
                    }

                    //puffInhalationDose += puffConc * totalInhalation * d_dt;
                    //puffCloudshineDose += puffConc * totalCloudshine * d_dt;

                    //if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0 && nuclideIdx == 0)
                    //    printf("%e, %e\n", sdata_inhalation[threadIdx.x]);
                }
            }

            //printf("[%d, %d] = %e\n", simIdx, evacueeIdx, puffCloudshineDose);

            //int pfidx = 0;
            //if (evacuee.flag == 0) pfidx = 1;
            //else if (evacuee.flag == 1) pfidx = 2;
            //else if (evacuee.flag == 2) pfidx = 0;

            //if(dPF->pfactor[pfidx][4]>0) puffInhalationDose *= dPF->pfactor[pfidx][4] * dPF->pfactor[pfidx][2];
            //if(dPF->pfactor[pfidx][0]>0) puffCloudshineDose *= dPF->pfactor[pfidx][0];

            //printf("%e\t%e\n", dPF->pfactor[pfidx][4], dPF->pfactor[pfidx][2]);

            //inhalationDose += puffInhalationDose * gaussianFactor;
            //cloudshineDose += puffCloudshineDose * gaussianFactor;

            //cloudshineDose += puffCloudshineDose * distanceFactor;

        }

        //float inhalationDose = sdata_inhalation[threadIdx.x];
        //float cloudshineDose = sdata_cloudshine[threadIdx.x];

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

        //if (inhalationDose > 0) sdata_inhalation[threadIdx.x] = inhalationDose;
        //else sdata_inhalation[threadIdx.x] = 0;

        //if (cloudshineDose > 0) sdata_cloudshine[threadIdx.x] = cloudshineDose;
        //else sdata_cloudshine[threadIdx.x] = 0;
        //__syncthreads();

        //for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        //    if (threadIdx.x < s) {
        //        sdata_inhalation[threadIdx.x] += sdata_inhalation[threadIdx.x + s];
        //        sdata_cloudshine[threadIdx.x] += sdata_cloudshine[threadIdx.x + s];
        //    }
        //    __syncthreads();
        //}

        //if (threadIdx.x == 0) {
        //    d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_inhalation += sdata_inhalation[threadIdx.x];
        //    d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_cloudshine += sdata_cloudshine[threadIdx.x];
        //}

        //if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 4)
        //    printf("%e\n", sdata_inhalation[threadIdx.x]);

        if (sdata_inhalation[threadIdx.x] > 0.000)
        //if (d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_inhalation > 0.000)
            printf("%d, %d\n", simIdx, evacueeIdx);

        //if (simIdx == 0 && evacueeIdx == 4 && puffIdx == 0) {
        //    printf("%e\n", d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_inhalation);
        //}

    }
}
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
    //float* sdata_cloudshine = sdata + blockDim.y;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim && puffIdx < d_totalpuff_per_Sim) {
        sdata_inhalation[threadIdx.x] = 0.0f;
        //sdata_cloudshine[threadIdx.x] = 0.0f;

        Gpuff::Puffcenter_RCAP puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];


        float inhalationDose = 1.0e-40;
        //float cloudshineDose = 0.0f;

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
        //if (1) {
            float puffInhalationDose = 1.0e-40;
            //float puffCloudshineDose = 0.0f;
            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffConc = puff.conc[nuclideIdx];// *gaussianFactor;

                if (puffConc > 0.0f) {
                    float totalInhalation = 1.0e-40;
                    //float totalCloudshine = 0.0f;

#pragma unroll
                    for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++) {

                        //float cloudshineValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 0];
                        //if (cloudshineValue > 0.0f) {
                        //    totalCloudshine += cloudshineValue;
                        //}

                        float inhalationValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2];
                        if (inhalationValue > 0.0f) {
                            totalInhalation += inhalationValue;
                        }
                    }

                    puffInhalationDose += puffConc * totalInhalation * d_dt;
                    //puffCloudshineDose += puffConc * totalCloudshine * d_dt;

                    //if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0 && nuclideIdx == 0)
                    //    printf("%e\n", sdata_inhalation[threadIdx.x]);
                }
            }

            if (dPF->pfactor[puff.flag][4] > 0) puffInhalationDose *= dPF->pfactor[puff.flag][4];
            //if (dPF->pfactor[puff.flag][2] > 0) puffCloudshineDose *= dPF->pfactor[puff.flag][2];

            inhalationDose += puffInhalationDose * gaussianFactor;
            //cloudshineDose += puffCloudshineDose * distanceFactor;
        }

        //float inhalationDose = sdata_inhalation[threadIdx.x];
        //float cloudshineDose = sdata_cloudshine[threadIdx.x];

        if (inhalationDose > 0) sdata_inhalation[threadIdx.x] = inhalationDose;
        else sdata_inhalation[threadIdx.x] = 1.0e-40;

        //if (cloudshineDose > 0) sdata_cloudshine[threadIdx.x] = cloudshineDose;
        //else sdata_cloudshine[threadIdx.x] = 0;
        __syncthreads();

        if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0) {
            // printf("%e\n", sdata_inhalation[threadIdx.x]);
        }

        //for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        //    inhalationDose += __shfl_down_sync(0xffffffff, inhalationDose, offset);
        //    cloudshineDose += __shfl_down_sync(0xffffffff, cloudshineDose, offset);
        //} 

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata_inhalation[threadIdx.x] += sdata_inhalation[threadIdx.x + s];
                //sdata_cloudshine[threadIdx.x] += sdata_cloudshine[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0) {
            //printf("%e\n", sdata_inhalation[threadIdx.x]);
        }

        if (threadIdx.x == 0) {
            //d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_inhalation = inhalationDose * gaussianFactor;
            //d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_cloudshine = cloudshineDose * distanceFactor;

            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_inhalation = sdata_inhalation[threadIdx.x];
            //d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_cloudshine = sdata_cloudshine[threadIdx.x];
        }

        //if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0)
        //    printf("%e\n", sdata_inhalation[threadIdx.x]);

        if (simIdx == 0 && evacueeIdx == 0 && puffIdx == 0) {
            //printf("%e\n", d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_inhalation);
        }

    }
}

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
    //float* sdata_cloudshine = sdata + blockDim.y;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim && puffIdx < d_totalpuff_per_Sim) {
        sdata_inhalation[threadIdx.x] = 0.0f;
        //sdata_cloudshine[threadIdx.x] = 0.0f;

        Gpuff::Puffcenter_RCAP puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

        if (puff.unitidx != 0) return;

        float inhalationDose = 1.0e-40;
        //float cloudshineDose = 0.0f;

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
            //if (1) {
            float puffInhalationDose = 1.0e-40;
            //float puffCloudshineDose = 0.0f;
            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffConc = puff.conc[nuclideIdx];// *gaussianFactor;

                if (puffConc > 0.0f) {
                    float totalInhalation = 1.0e-40;
                    //float totalCloudshine = 0.0f;

#pragma unroll
                    for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++) {

                        //float cloudshineValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 0];
                        //if (cloudshineValue > 0.0f) {
                        //    totalCloudshine += cloudshineValue;
                        //}

                        float inhalationValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2];
                        if (inhalationValue > 0.0f) {
                            totalInhalation += inhalationValue;
                        }
                    }

                    puffInhalationDose += puffConc * totalInhalation * d_dt;
                    //puffCloudshineDose += puffConc * totalCloudshine * d_dt;

                    //if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0 && nuclideIdx == 0)
                    //    printf("%e\n", sdata_inhalation[threadIdx.x]);
                }
            }

            if (dPF->pfactor[puff.flag][4] > 0) puffInhalationDose *= dPF->pfactor[puff.flag][4];
            //if (dPF->pfactor[puff.flag][2] > 0) puffCloudshineDose *= dPF->pfactor[puff.flag][2];

            inhalationDose += puffInhalationDose * gaussianFactor;
            //cloudshineDose += puffCloudshineDose * distanceFactor;
        }

        //float inhalationDose = sdata_inhalation[threadIdx.x];
        //float cloudshineDose = sdata_cloudshine[threadIdx.x];

        if (inhalationDose > 0) sdata_inhalation[threadIdx.x] = inhalationDose;
        else sdata_inhalation[threadIdx.x] = 1.0e-40;

        //if (cloudshineDose > 0) sdata_cloudshine[threadIdx.x] = cloudshineDose;
        //else sdata_cloudshine[threadIdx.x] = 0;
        __syncthreads();

        if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0) {
            // printf("%e\n", sdata_inhalation[threadIdx.x]);
        }

        //for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        //    inhalationDose += __shfl_down_sync(0xffffffff, inhalationDose, offset);
        //    cloudshineDose += __shfl_down_sync(0xffffffff, cloudshineDose, offset);
        //} 

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata_inhalation[threadIdx.x] += sdata_inhalation[threadIdx.x + s];
                //sdata_cloudshine[threadIdx.x] += sdata_cloudshine[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0) {
            //printf("%e\n", sdata_inhalation[threadIdx.x]);
        }

        if (threadIdx.x == 0) {
            //d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_inhalation = inhalationDose * gaussianFactor;
            //d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_cloudshine = cloudshineDose * distanceFactor;

            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_cloudshine = 6.0 * sdata_inhalation[threadIdx.x];
            //d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_cloudshine = sdata_cloudshine[threadIdx.x];
        }

        //if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0)
        //    printf("%e\n", sdata_inhalation[threadIdx.x]);

        if (simIdx == 0 && evacueeIdx == 0 && puffIdx == 0) {
            //printf("%e\n", d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_inhalation);
        }

    }
}

__global__ void ComputeExposureHmix_1D(
    Gpuff::Puffcenter_RCAP* __restrict__ d_puffs_RCAP,
    Evacuee* __restrict__ d_evacuees,
    const float* __restrict__ d_exposure,
    const ProtectionFactors* __restrict__ dPF
) {
    extern __shared__ float sdata[];

    // 1D optimization: convert all simulation parameters to a single global index
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate required index ranges: decompose simIdx, evacueeIdx, puffIdx from global index
    int numEvacuees = d_totalevacuees_per_Sim * d_numSims;
    int numPuffs = d_totalpuff_per_Sim * d_numSims;

    if (globalIdx >= numEvacuees * numPuffs) {
        return; // Return early if index exceeds valid range
    }

    int simIdx = globalIdx / (d_totalevacuees_per_Sim * d_totalpuff_per_Sim);
    int evacueeIdx = (globalIdx / d_totalpuff_per_Sim) % d_totalevacuees_per_Sim;
    int puffIdx = globalIdx % d_totalpuff_per_Sim;

    float hmix = 1500.0;
    float* sdata_inhalation = sdata;
    float* sdata_cloudshine = sdata + blockDim.x;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim && puffIdx < d_totalpuff_per_Sim) {
        sdata_inhalation[threadIdx.x] = 0.0f;
        sdata_cloudshine[threadIdx.x] = 0.0f;

        Gpuff::Puffcenter_RCAP puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

        float cosTheta = __cosf(evacuee.theta);
        float sinTheta = __sinf(evacuee.theta);

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

        if (1) {
            float puffInhalationDose = 0.0f;
            float puffCloudshineDose = 0.0f;
            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                float puffConc = puff.conc[nuclideIdx];

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

                    puffInhalationDose += puffConc * totalInhalation * d_dt;
                    puffCloudshineDose += puffConc * totalCloudshine * d_dt;
                }
            }

            if (dPF->pfactor[puff.flag][4] > 0) puffInhalationDose *= dPF->pfactor[puff.flag][4];
            if (dPF->pfactor[puff.flag][2] > 0) puffCloudshineDose *= dPF->pfactor[puff.flag][2];

            inhalationDose += puffInhalationDose * gaussianFactor;
            cloudshineDose += puffCloudshineDose * distanceFactor;
        }

        if (inhalationDose > 0) sdata_inhalation[threadIdx.x] = inhalationDose;
        else sdata_inhalation[threadIdx.x] = 0;

        if (cloudshineDose > 0) sdata_cloudshine[threadIdx.x] = cloudshineDose;
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
            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_inhalation = sdata_inhalation[threadIdx.x];
        }
    }
}


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

                //printf("sigmah = %e\n", puffs_RCAP[simIdx * totalPuffsPerSim + puffIdx].sigma_h);

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

                //printf("sigma_h = %e, sigma_z = %e, dx = %e, dy = %e\n", sigma_h, sigma_z, dx, dy);
                //printf("gaussianFactor = %e\n", gaussianFactor);



                float distanceFactor = 1 / (4.0f * PI * (dx * dx + dy * dy + H * H));

                float inhalationDose = 0.0f;
                float cloudshineDose = 0.0f;

                //if (gaussianFactor > 1e-30f) {
                if (1) {
                    //float inhalationDose = 0.0f;
                    //float cloudshineDose = 0.0f;

                    for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                        float puffConc = puff.conc[nuclideIdx];
                        //printf("%e\n", puffConc);

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

                                //printf("cl = %e, ih = %e\n", cloudshineValue, inhalationValue);
                                //printf("cl = %e\n", exposure_data_all[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 0]);

                            }

                            inhalationDose += puffConc * totalInhalation * dt;
                            cloudshineDose += puffConc * totalCloudshine * dt;

                            //if (simIdx == 0 && puffIdx == 0 && nuclideIdx == 0 && evacueeIdx == 0)
                            //    printf("%e\n", inhalationDose);
                        }
                    }

                    inhalationDose *= PF.pfactor[puff.flag][4];
                    cloudshineDose *= PF.pfactor[puff.flag][2];

                    printf("flag=%d, factor=%f\n", puff.flag, PF.pfactor[puff.flag][4]);
                    //printf("PF.[4] = %f, PF.[2] = %f\n", cloudshineValue, inhalationValue);

                    //evacuees[simIdx * totalEvacueesPerSim + evacueeIdx].dose_inhalation += inhalationDose * gaussianFactor;
                    //evacuees[simIdx * totalEvacueesPerSim + evacueeIdx].dose_cloudshine += cloudshineDose * distanceFactor;
                    
                    //if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0)
                    //    printf("%e\n", inhalationDose);

                    //inhal += inhalationDose * gaussianFactor;
                    //cshine += cloudshineDose * distanceFactor;
                    //printf("cshine = %f\n", cshine);

                    //if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0)
                    //    printf("%e\n", inhalationDose * gaussianFactor);

                }
                //if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0)
                //    printf("%e\n", inhal);
                sum_inhal += inhalationDose * gaussianFactor;
                sum_cshine += cloudshineDose * distanceFactor;
                //if (simIdx == 0 && puffIdx == 0 && evacueeIdx == 0)
                //    printf("%e\n", cshine);
            }

            //if (simIdx == 0 && evacueeIdx == 0) {
            //    printf("%e\n", sum_inhal);
            //}

            //if(sum_inhal>0.001) printf("sum=%d, eva=%d\n", simIdx, evacueeIdx);




            evacuees[simIdx * totalEvacueesPerSim + evacueeIdx].dose_inhalation = sum_inhal;
            evacuees[simIdx * totalEvacueesPerSim + evacueeIdx].dose_cloudshine = sum_cshine;

            if (simIdx == 0 && evacueeIdx == 4) {
                printf("%e\n", evacuees[simIdx * totalevacuees_per_Sim + evacueeIdx].dose_inhalation);
            }
        }
    }
}

__global__ void ComputeExposureHmix1(
    Gpuff::Puffcenter_RCAP* __restrict__ d_puffs_RCAP,
    Evacuee* __restrict__ d_evacuees,
    const float* __restrict__ d_exposure,
    const ProtectionFactors* __restrict__ dPF
) {
    extern __shared__ float sdata[];

    int simIdx = blockIdx.x;
    int evacueeIdx = blockIdx.y;
    int puffIdx = threadIdx.x;

    float hmix = 1500.0f;

    float* sdata_inhalation = sdata;
    float* sdata_cloudshine = sdata + blockDim.x;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim && puffIdx < d_totalpuff_per_Sim) {
        sdata_inhalation[threadIdx.x] = 0.0f;
        sdata_cloudshine[threadIdx.x] = 0.0f;

        Gpuff::Puffcenter_RCAP puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
        Evacuee evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

        float cosTheta = __cosf(evacuee.theta);
        float sinTheta = __sinf(evacuee.theta);

        // Position calculation
        float dx = evacuee.r * cosTheta - puff.x;
        float dy = evacuee.r * sinTheta - puff.y;
        float z_evac = 0.0f; // Evacuee height (ground level)
        float H = puff.z;     // Puff center height

        float sigma_h = puff.sigma_h;
        float sigma_z = puff.sigma_z;

        // Distance squared calculation (3D distance including vertical separation)
        float distanceSq = dx * dx + dy * dy + (H - z_evac) * (H - z_evac);

        // Gaussian plume calculation for inhalation (ground reflection and mixing height reflection)
        float sumExponent = 0.0f;
        int nLimit = 3;
        for (int n = -nLimit; n <= nLimit; ++n) {
            float H_n = H + 2.0f * n * hmix;
            float z_diff1 = z_evac - H_n;
            float z_diff2 = z_evac + H_n;

            float exponent1 = -(dx * dx + dy * dy) / (2.0f * sigma_h * sigma_h)
                - (z_diff1 * z_diff1) / (2.0f * sigma_z * sigma_z);
            float exponent2 = -(dx * dx + dy * dy) / (2.0f * sigma_h * sigma_h)
                - (z_diff2 * z_diff2) / (2.0f * sigma_z * sigma_z);

            sumExponent += __expf(exponent1) + __expf(exponent2);
        }

        float gaussianFactor = sumExponent / (2.0f * PI * sigma_h * sigma_z);

        // Distance calculation for CloudShine
        float distance = sqrtf(distanceSq);
        float pointSourceFactor = 1.0f / (4.0f * PI * distanceSq); // Point source approximation

        // Check if concentration threshold is exceeded
        if (gaussianFactor > 1e-30f || pointSourceFactor > 1e-30f) {
            for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
                // Inhalation dose calculation
                float puffConcInhalation = puff.conc[nuclideIdx] * gaussianFactor;

                // CloudShine intensity calculation (point source approximation)
                float puffIntensityCloudshine = puff.conc[nuclideIdx] * pointSourceFactor;

                float totalInhalation = 0.0f;
                float totalCloudshine = 0.0f;

#pragma unroll
                for (int organIdx = 0; organIdx < MAX_ORGANS; organIdx++) {
                    // Inhalation dose (DATA_FIELDS + 2)
                    float inhalationValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2];
                    if (inhalationValue > 0.0f) {
                        totalInhalation += inhalationValue;
                    }

                    // CloudShine dose (DATA_FIELDS + 0)
                    float cloudshineValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 0];
                    if (cloudshineValue > 0.0f) {
                        totalCloudshine += cloudshineValue;
                    }
                }

                // Accumulate dose with time step
                sdata_inhalation[threadIdx.x] += puffConcInhalation * totalInhalation * d_dt;
                sdata_cloudshine[threadIdx.x] += puffIntensityCloudshine * totalCloudshine * d_dt;
            }

            // Apply protection factors
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


__global__ void DirectInhalation2(
    const Gpuff::Puffcenter_RCAP* __restrict__ d_puffs_RCAP,
    Evacuee* __restrict__ d_evacuees,
    const float* __restrict__ d_exposure,
    const ProtectionFactors* __restrict__ dPF
) {
    extern __shared__ float sdata[];

    int simIdx = blockIdx.x;
    int evacueeIdx = blockIdx.y;
    int puffIdx = threadIdx.x;
    int organIdx = threadIdx.y;

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim) {
        sdata[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;

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
                    float exposureValue = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS + organIdx * DATA_FIELDS + 2];

                    float totalExposure = 0.0f;
                    if (exposureValue > 0.0f) {
                        totalExposure = exposureValue;
                    }

                    sdata[threadIdx.y * blockDim.x + threadIdx.x] += puffEffect * totalExposure * d_dt;
                }
            }
        }

        sdata[threadIdx.y * blockDim.x + threadIdx.x] *= dPF->pfactor[puff.flag][4] * dPF->pfactor[puff.flag][2];

        __syncthreads();

        float val = sdata[threadIdx.y * blockDim.x + threadIdx.x];
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                sdata[threadIdx.y * blockDim.x + threadIdx.x] += sdata[threadIdx.y * blockDim.x + threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float organVal = sdata[threadIdx.y * blockDim.x];
            for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
                if (threadIdx.y < offset) {
                    sdata[threadIdx.y * blockDim.x] += sdata[(threadIdx.y + offset) * blockDim.x];
                }
                __syncthreads();
            }

            if (threadIdx.y == 0) {
                d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose = sdata[0];
            }
        }
    }
}




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
            //printf("doseSum = %f\n", doseSum);
            //sdata[threadIdx.x] = doseSum * d_dt;
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

__global__ void computeEvacueeDoseReductionFlat(
    Gpuff::Puffcenter_RCAP* d_puffs_RCAP,
    Evacuee* d_evacuees, float* d_exposure
) {
    extern __shared__ float sdata[];

    int simIdx = blockIdx.x;
    int evacueeIdx = blockIdx.y;
    int threadId = threadIdx.x;
    int puffIdx = threadId / MAX_ORGANS;  // Compute puff index from flattened index
    int organIdx = threadId % MAX_ORGANS; // Compute organ index from flattened index

    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim && puffIdx < d_totalpuff_per_Sim) {
        sdata[threadId] = 0.0f;

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
            sdata[threadId] = doseSum * d_dt;
        }

        __syncthreads();

        // Perform parallel reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadId < s) {
                sdata[threadId] += sdata[threadId + s];
            }
            __syncthreads();
        }

        // Write result for this block to global mem
        if (threadId == 0) {
            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose += sdata[0];
        }
    }
}

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



//__global__ void computeEvacueeDoseReduction(
//    Gpuff::Puffcenter_RCAP* d_puffs_RCAP,
//    Evacuee* d_evacuees,
//    NuclideData* d_ND
//) {
//    extern __shared__ float sdata[];
//
//    int simIdx = blockIdx.x;
//    int evacueeIdx = blockIdx.y; 
//    int puffIdx = threadIdx.x;
//
//    if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim) {
//        sdata[threadIdx.x] = 0.0f;
//
//        Gpuff::Puffcenter_RCAP& puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
//        Evacuee& evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];
//
//        float dx = evacuee.r * cos(evacuee.theta) - puff.x;
//        float dy = evacuee.r * sin(evacuee.theta) - puff.y;
//        float distance = sqrt(dx * dx + dy * dy);
//
//        if (distance > 0.0f) {
//
//
//            for (int nucIdx = 0; nucIdx < MAX_NUCLIDES; ++nucIdx) {
//                float puffEffect = puff.conc[nucIdx] / (distance * distance);
//                //printf("d_ND[%d].exposure_data[organIdx][1] = %f\n", nucIdx, d_ND[0].exposure_data[0][0]);
//                sdata[threadIdx.x] += puffEffect * d_dt;
//
//                //for (int organIdx = 0; organIdx < d_ND[nucIdx].organ_count; ++organIdx) {
//                //    float dcf = d_ND[nucIdx].exposure_data[organIdx][1];
//                //    sdata[threadIdx.x] += puffEffect * dcf * d_dt;
//                //}
//            }
//        }
//
//        __syncthreads();
//
//        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//            if (threadIdx.x < s) {
//                sdata[threadIdx.x] += sdata[threadIdx.x + s];
//            }
//            __syncthreads();
//        }
//
//        if (threadIdx.x == 0) {
//            d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose = sdata[0];
//        }
//    }
//}

//__global__ void printExposureData(NuclideData* d_ND) {
//    for (int i = 0; i < MAX_NUCLIDES; ++i) {
//        printf("Nuclide %d:\n", i);
//        for (int j = 0; j < MAX_ORGANS; ++j) {
//            printf("  Organ %d:\n", j);
//            for (int k = 0; k < DATA_FIELDS; ++k) {
//                printf("    Exposure Data[%d][%d]: %f\n", j, k, d_ND[i].exposure_data[j * DATA_FIELDS + k]);
//            }
//        }
//    }
//}

//__global__ void printExposureData(float* d_exposure_data) {
//
//    if (i < MAX_NUCLIDES && j < MAX_ORGANS) {
//        printf("Nuclide %d:\n", i);
//        printf("  Organ %d:\n", j);
//        for (int k = 0; k < DATA_FIELDS; ++k) {
//            int idx = i * (MAX_ORGANS * DATA_FIELDS) + j * DATA_FIELDS + k;
//            printf("    Exposure Data[%d][%d]: %f\n", j, k, d_exposure_data[idx]);
//        }
//    }
//}


//__global__ void printExposureData(float* d_exposure_data) {
//    for (int i = 0; i < MAX_NUCLIDES; ++i) {
//        printf("Nuclide %d:\n", i);
//        for (int j = 0; j < MAX_ORGANS; ++j) {
//            printf("  Organ %d:\n", j);
//            for (int k = 0; k < DATA_FIELDS; ++k) {
//                int idx = i * (MAX_ORGANS * DATA_FIELDS) + j * DATA_FIELDS + k;
//                printf("    Exposure Data[%d][%d]: %e\n", j, k, d_exposure_data[idx]);
//            }
//        }
//    }
//}

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