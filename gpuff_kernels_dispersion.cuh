// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel Implementation - Dispersion Functions
// ====================================================================================
//
// File: gpuff_kernels_dispersion.cuh
// Purpose: Atmospheric dispersion formulas and helper functions for GPUFF
//
// This file contains:
//   - Atomic operations for GPU thread synchronization
//   - Pasquill-Gifford dispersion formulas
//   - Briggs-McElroy-Pooler dispersion formulas
//   - Newton-Raphson iteration for virtual distance calculation
//
// ====================================================================================

#ifndef GPUFF_KERNELS_DISPERSION_CUH
#define GPUFF_KERNELS_DISPERSION_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

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

// Note: Device variables d_isPG and d_isRural are defined in gpuff.cuh
// No extern declarations needed as they're accessible through the include chain

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

#endif // GPUFF_KERNELS_DISPERSION_CUH