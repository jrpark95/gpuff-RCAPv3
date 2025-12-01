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
 * Tadmor-Gur (MACCS Table 2-4) Power Law sigma_y calculation
 * Formula: sigma_y = a_y * x^b_y
 * Reference: MACCS Manual Table 2-4, Tadmor & Gur (1969)
 */
float Sigma_y_TadmorGur_cpu(int PasquillCategory, float distance) {
    // Tadmor-Gur coefficients (Table 2-4)
    // Index: A=0, B=1, C=2, D=3, E=4, F=5, G=6
    float a_y[7] = {0.3658f, 0.2751f, 0.2089f, 0.1474f, 0.1046f, 0.0722f, 0.0722f};
    float b_y[7] = {0.9031f, 0.9031f, 0.9031f, 0.9031f, 0.9031f, 0.9031f, 0.9031f};

    if (distance < 1.0f) distance = 1.0f;  // Minimum distance
    float sigma = a_y[PasquillCategory] * powf(distance, b_y[PasquillCategory]);
    return sigma;
}

/**
 * Tadmor-Gur (MACCS Table 2-4) Power Law sigma_z calculation
 * Formula: sigma_z = a_z * x^b_z
 * Reference: MACCS Manual Table 2-4, Tadmor & Gur (1969)
 */
float Sigma_z_TadmorGur_cpu(int PasquillCategory, float distance) {
    // Tadmor-Gur coefficients (Table 2-4)
    // Index: A=0, B=1, C=2, D=3, E=4, F=5, G=6
    float a_z[7] = {0.00025f, 0.0019f, 0.2f, 0.3f, 0.4f, 0.2f, 0.2f};
    float b_z[7] = {2.125f, 1.6021f, 0.8543f, 0.6532f, 0.6021f, 0.6020f, 0.6020f};

    if (distance < 1.0f) distance = 1.0f;  // Minimum distance
    float sigma = a_z[PasquillCategory] * powf(distance, b_z[PasquillCategory]);
    return sigma;
}

/**
 * NUREG/CR-7161 (MACCS Table 2-5) Power Law sigma_y calculation
 * Formula: sigma_y = a_y * x^b_y
 * Reference: MACCS Manual Table 2-5, Expert Elicitation (Bixler et al., 2013)
 */
float Sigma_y_NUREG7161_cpu(int PasquillCategory, float distance) {
    // NUREG/CR-7161 coefficients (Table 2-5)
    // Note: A/B share same values, E/F share same values
    // Index: A=0, B=1, C=2, D=3, E=4, F=5, G=6
    float a_y[7] = {0.7507f, 0.7507f, 0.4063f, 0.2779f, 0.2158f, 0.2158f, 0.2158f};
    float b_y[7] = {0.866f, 0.866f, 0.865f, 0.881f, 0.866f, 0.866f, 0.866f};

    if (distance < 1.0f) distance = 1.0f;  // Minimum distance
    float sigma = a_y[PasquillCategory] * powf(distance, b_y[PasquillCategory]);
    return sigma;
}

/**
 * NUREG/CR-7161 (MACCS Table 2-5) Power Law sigma_z calculation
 * Formula: sigma_z = a_z * x^b_z
 * Reference: MACCS Manual Table 2-5, Expert Elicitation (Bixler et al., 2013)
 */
float Sigma_z_NUREG7161_cpu(int PasquillCategory, float distance) {
    // NUREG/CR-7161 coefficients (Table 2-5)
    // Note: A/B share same values, E/F share same values
    // Index: A=0, B=1, C=2, D=3, E=4, F=5, G=6
    float a_z[7] = {0.0361f, 0.0361f, 0.2036f, 0.2636f, 0.2463f, 0.2463f, 0.2463f};
    float b_z[7] = {1.277f, 1.277f, 0.859f, 0.751f, 0.619f, 0.619f, 0.619f};

    if (distance < 1.0f) distance = 1.0f;  // Minimum distance
    float sigma = a_z[PasquillCategory] * powf(distance, b_z[PasquillCategory]);
    return sigma;
}

// ====================================================================================
// GPU Device Functions - Tadmor-Gur and NUREG/CR-7161 Dispersion
// ====================================================================================

/**
 * GPU Device: Tadmor-Gur sigma_y calculation
 */
__device__ float Sigma_y_TadmorGur(int PasquillCategory, float distance) {
    float a_y[7] = {0.3658f, 0.2751f, 0.2089f, 0.1474f, 0.1046f, 0.0722f, 0.0722f};
    float b_y[7] = {0.9031f, 0.9031f, 0.9031f, 0.9031f, 0.9031f, 0.9031f, 0.9031f};

    if (distance < 1.0f) distance = 1.0f;
    return a_y[PasquillCategory] * powf(distance, b_y[PasquillCategory]);
}

/**
 * GPU Device: Tadmor-Gur sigma_z calculation
 */
__device__ float Sigma_z_TadmorGur(int PasquillCategory, float distance) {
    float a_z[7] = {0.00025f, 0.0019f, 0.2f, 0.3f, 0.4f, 0.2f, 0.2f};
    float b_z[7] = {2.125f, 1.6021f, 0.8543f, 0.6532f, 0.6021f, 0.6020f, 0.6020f};

    if (distance < 1.0f) distance = 1.0f;
    return a_z[PasquillCategory] * powf(distance, b_z[PasquillCategory]);
}

/**
 * GPU Device: NUREG/CR-7161 sigma_y calculation
 */
__device__ float Sigma_y_NUREG7161(int PasquillCategory, float distance) {
    float a_y[7] = {0.7507f, 0.7507f, 0.4063f, 0.2779f, 0.2158f, 0.2158f, 0.2158f};
    float b_y[7] = {0.866f, 0.866f, 0.865f, 0.881f, 0.866f, 0.866f, 0.866f};

    if (distance < 1.0f) distance = 1.0f;
    return a_y[PasquillCategory] * powf(distance, b_y[PasquillCategory]);
}

/**
 * GPU Device: NUREG/CR-7161 sigma_z calculation
 */
__device__ float Sigma_z_NUREG7161(int PasquillCategory, float distance) {
    float a_z[7] = {0.0361f, 0.0361f, 0.2036f, 0.2636f, 0.2463f, 0.2463f, 0.2463f};
    float b_z[7] = {1.277f, 1.277f, 0.859f, 0.751f, 0.619f, 0.619f, 0.619f};

    if (distance < 1.0f) distance = 1.0f;
    return a_z[PasquillCategory] * powf(distance, b_z[PasquillCategory]);
}

// ====================================================================================
// Hybrid T-G Modified Functions (Tadmor-Gur < 5km, NUREG/CR-7161 >= 5km)
// Uses virtual source concept for continuity at transition
// ====================================================================================

// Transition distance (meters)
constexpr float HYBRID_TRANSITION_DISTANCE = 5000.0f;

/**
 * GPU Device: Hybrid sigma_y (T-G Modified)
 * - Uses Tadmor-Gur for distance < 5km
 * - Uses NUREG/CR-7161 for distance >= 5km with virtual source for continuity
 */
__device__ float Sigma_y_Hybrid(int PasquillCategory, float distance) {
    if (distance < 1.0f) distance = 1.0f;

    if (distance < HYBRID_TRANSITION_DISTANCE) {
        // Use Tadmor-Gur below 5km
        return Sigma_y_TadmorGur(PasquillCategory, distance);
    } else {
        // Calculate sigma at transition point using Tadmor-Gur
        float sigma_at_transition = Sigma_y_TadmorGur(PasquillCategory, HYBRID_TRANSITION_DISTANCE);

        // Calculate virtual source distance for NUREG/CR-7161
        // sigma = a * x^b => x = (sigma/a)^(1/b)
        float a_y[7] = {0.7507f, 0.7507f, 0.4063f, 0.2779f, 0.2158f, 0.2158f, 0.2158f};
        float b_y[7] = {0.866f, 0.866f, 0.865f, 0.881f, 0.866f, 0.866f, 0.866f};

        float virtual_distance_at_transition = powf(sigma_at_transition / a_y[PasquillCategory], 1.0f / b_y[PasquillCategory]);

        // Calculate effective distance from virtual source
        float distance_beyond_transition = distance - HYBRID_TRANSITION_DISTANCE;
        float effective_distance = virtual_distance_at_transition + distance_beyond_transition;

        return Sigma_y_NUREG7161(PasquillCategory, effective_distance);
    }
}

/**
 * GPU Device: Hybrid sigma_z (T-G Modified)
 * - Uses Tadmor-Gur for distance < 5km
 * - Uses NUREG/CR-7161 for distance >= 5km with virtual source for continuity
 */
__device__ float Sigma_z_Hybrid(int PasquillCategory, float distance) {
    if (distance < 1.0f) distance = 1.0f;

    if (distance < HYBRID_TRANSITION_DISTANCE) {
        // Use Tadmor-Gur below 5km
        return Sigma_z_TadmorGur(PasquillCategory, distance);
    } else {
        // Calculate sigma at transition point using Tadmor-Gur
        float sigma_at_transition = Sigma_z_TadmorGur(PasquillCategory, HYBRID_TRANSITION_DISTANCE);

        // Calculate virtual source distance for NUREG/CR-7161
        float a_z[7] = {0.0361f, 0.0361f, 0.2036f, 0.2636f, 0.2463f, 0.2463f, 0.2463f};
        float b_z[7] = {1.277f, 1.277f, 0.859f, 0.751f, 0.619f, 0.619f, 0.619f};

        float virtual_distance_at_transition = powf(sigma_at_transition / a_z[PasquillCategory], 1.0f / b_z[PasquillCategory]);

        // Calculate effective distance from virtual source
        float distance_beyond_transition = distance - HYBRID_TRANSITION_DISTANCE;
        float effective_distance = virtual_distance_at_transition + distance_beyond_transition;

        return Sigma_z_NUREG7161(PasquillCategory, effective_distance);
    }
}

/**
 * CPU: Hybrid sigma_y (T-G Modified) - for validation
 */
float Sigma_y_Hybrid_cpu(int PasquillCategory, float distance) {
    if (distance < 1.0f) distance = 1.0f;

    if (distance < HYBRID_TRANSITION_DISTANCE) {
        return Sigma_y_TadmorGur_cpu(PasquillCategory, distance);
    } else {
        float sigma_at_transition = Sigma_y_TadmorGur_cpu(PasquillCategory, HYBRID_TRANSITION_DISTANCE);

        float a_y[7] = {0.7507f, 0.7507f, 0.4063f, 0.2779f, 0.2158f, 0.2158f, 0.2158f};
        float b_y[7] = {0.866f, 0.866f, 0.865f, 0.881f, 0.866f, 0.866f, 0.866f};

        float virtual_distance_at_transition = powf(sigma_at_transition / a_y[PasquillCategory], 1.0f / b_y[PasquillCategory]);
        float distance_beyond_transition = distance - HYBRID_TRANSITION_DISTANCE;
        float effective_distance = virtual_distance_at_transition + distance_beyond_transition;

        return Sigma_y_NUREG7161_cpu(PasquillCategory, effective_distance);
    }
}

/**
 * CPU: Hybrid sigma_z (T-G Modified) - for validation
 */
float Sigma_z_Hybrid_cpu(int PasquillCategory, float distance) {
    if (distance < 1.0f) distance = 1.0f;

    if (distance < HYBRID_TRANSITION_DISTANCE) {
        return Sigma_z_TadmorGur_cpu(PasquillCategory, distance);
    } else {
        float sigma_at_transition = Sigma_z_TadmorGur_cpu(PasquillCategory, HYBRID_TRANSITION_DISTANCE);

        float a_z[7] = {0.0361f, 0.0361f, 0.2036f, 0.2636f, 0.2463f, 0.2463f, 0.2463f};
        float b_z[7] = {1.277f, 1.277f, 0.859f, 0.751f, 0.619f, 0.619f, 0.619f};

        float virtual_distance_at_transition = powf(sigma_at_transition / a_z[PasquillCategory], 1.0f / b_z[PasquillCategory]);
        float distance_beyond_transition = distance - HYBRID_TRANSITION_DISTANCE;
        float effective_distance = virtual_distance_at_transition + distance_beyond_transition;

        return Sigma_z_NUREG7161_cpu(PasquillCategory, effective_distance);
    }
}

/**
 * Calculate Plume Rise using Briggs equations (MACCS model)
 *
 * @param rel_heat  Heat release rate (W)
 * @param windspeed Wind speed at release height (m/s)
 * @param stability Pasquill stability class (0=A, 1=B, ..., 5=F, 6=G)
 * @return delta_h  Plume rise (m)
 *
 * Reference: MACCS Manual Section 2.4, Briggs (1969, 1975)
 */
float calculate_plume_rise(float rel_heat, float windspeed, int stability) {
    // Minimum windspeed to avoid division by zero
    float u = fmaxf(windspeed, 0.5f);

    // Buoyancy flux: F = 8.79e-6 * Q (W)
    float F = 8.79e-6f * rel_heat;

    // If no heat release, no plume rise
    if (F <= 0.0f) return 0.0f;

    float delta_h = 0.0f;

    if (stability <= 3) {
        // Unstable/Neutral conditions (A, B, C, D)
        if (F >= 55.0f) {
            delta_h = 38.7f * powf(F, 0.6f) / u;
        } else {
            delta_h = 21.4f * powf(F, 0.75f) / u;
        }
    } else {
        // Stable conditions (E, F, G)
        // Stability parameter S (s^-2)
        float S;
        if (stability == 4) {
            S = 5.04e-4f;  // E class
        } else {
            S = 1.27e-3f;  // F, G class
        }
        delta_h = 2.4f * powf(F / (u * S), 1.0f/3.0f);
    }

    return delta_h;
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
__device__ float Sigma_h_Briggs_McElroy_Pooler(int PasquillCategory, float virtual_distance){

    float coefficient0_rural[7] = {0.22, 0.16, 0.11, 0.08, 0.06, 0.04, 0.04};
    float coefficient0_urban[7] = {0.32, 0.32, 0.22, 0.16, 0.11, 0.11, 0.11};
    float coefficient1_rural = 0.0001;
    float coefficient1_urban = 0.0004;

    float sigma;
    
    if(d_isRural) sigma = coefficient0_rural[PasquillCategory]*virtual_distance
                            *pow(1 + coefficient1_rural*virtual_distance, -0.5);

    else sigma = coefficient0_urban[PasquillCategory]*virtual_distance
                    *pow(1 + coefficient1_urban*virtual_distance, -0.5);

    return sigma;
}

__device__ float dSh_BMP(int PasquillCategory, float virtual_distance){

    float coefficient0_rural[7] = {0.22, 0.16, 0.11, 0.08, 0.06, 0.04, 0.04};
    float coefficient0_urban[7] = {0.32, 0.32, 0.22, 0.16, 0.11, 0.11, 0.11};
    float coefficient1_rural = 0.0001;
    float coefficient1_urban = 0.0004;

    float sigma;
    
    if(d_isRural) sigma = 0.5*coefficient0_rural[PasquillCategory]
                            *(coefficient1_rural*virtual_distance+2)
                            /pow(coefficient1_rural*virtual_distance+1,1.5);

    else sigma = 0.5*coefficient0_urban[PasquillCategory]
                    *(coefficient1_urban*virtual_distance+2)
                    /pow(coefficient1_urban*virtual_distance+1,1.5);

    return sigma;
}

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
// CPU Helper Functions - Briggs-McElroy-Pooler Dispersion Formulas
// ====================================================================================

// Global variable for CPU terrain type (matches device d_isRural)
static bool cpu_isRural = true;

/**
 * CPU version: Briggs-McElroy-Pooler horizontal dispersion coefficient
 */
float Sigma_h_Briggs_McElroy_Pooler_cpu(int PasquillCategory, float virtual_distance) {
    float coefficient0_rural[7] = {0.22f, 0.16f, 0.11f, 0.08f, 0.06f, 0.04f, 0.04f};
    float coefficient0_urban[7] = {0.32f, 0.32f, 0.22f, 0.16f, 0.11f, 0.11f, 0.11f};
    float coefficient1_rural = 0.0001f;
    float coefficient1_urban = 0.0004f;

    float sigma;

    if (cpu_isRural) {
        sigma = coefficient0_rural[PasquillCategory] * virtual_distance
                * powf(1 + coefficient1_rural * virtual_distance, -0.5f);
    } else {
        sigma = coefficient0_urban[PasquillCategory] * virtual_distance
                * powf(1 + coefficient1_urban * virtual_distance, -0.5f);
    }

    return sigma;
}

/**
 * CPU version: Briggs-McElroy-Pooler vertical dispersion coefficient
 */
float Sigma_z_Briggs_McElroy_Pooler_cpu(int PasquillCategory, float virtual_distance) {
    float coefficient0_rural[7] = {0.20f, 0.12f, 0.08f, 0.06f, 0.03f, 0.016f, 0.016f};
    float coefficient1_rural[7] = {0.0f, 0.0f, 0.0002f, 0.0015f, 0.0003f, 0.0003f, 0.0003f};
    float coefficient2_rural[7] = {1.0f, 1.0f, -0.5f, -0.5f, -1.0f, -1.0f, -1.0f};

    float coefficient0_urban[7] = {0.24f, 0.24f, 0.2f, 0.14f, 0.08f, 0.08f, 0.08f};
    float coefficient1_urban[7] = {0.001f, 0.001f, 0.0f, 0.0003f, 0.00015f, 0.00015f, 0.00015f};
    float coefficient2_urban[7] = {0.5f, 0.5f, 1.0f, -0.5f, -0.5f, -0.5f, -0.5f};

    float sigma;

    if (cpu_isRural) {
        sigma = coefficient0_rural[PasquillCategory] * virtual_distance *
                powf(1 + coefficient1_rural[PasquillCategory] * virtual_distance, coefficient2_rural[PasquillCategory]);
    } else {
        sigma = coefficient0_urban[PasquillCategory] * virtual_distance *
                powf(1 + coefficient1_urban[PasquillCategory] * virtual_distance, coefficient2_urban[PasquillCategory]);
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
        else {
            fx = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, x) - targetSigma;
            dfx = dSh_BMP(PasquillCategory, x);
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