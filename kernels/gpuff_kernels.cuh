// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernels - Main Include File
// ====================================================================================
//
// File: gpuff_kernels.cuh
// Purpose: Central include file for all CUDA kernel modules
//
// This file organizes and includes all modularized kernel components:
//   - Constants and helper functions
//   - Transport kernels
//   - Deposition and decay kernels
//   - Dispersion kernels
//   - Concentration calculations
//   - Evacuation simulations
//   - Exposure calculations
//   - Utility functions
//
// ====================================================================================

#ifndef GPUFF_KERNELS_CUH
#define GPUFF_KERNELS_CUH

// Include all necessary CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cmath>
#include <float.h>

// Include project headers
#include "../gpuff.h"
#include "../Dose.cuh"
#include "../Evacuee_RCAP.cuh"
#include "../EvacuationDirections.h"

// ====================================================================================
// Module Includes
// ====================================================================================

// 1. Constants and helper functions (base module)
#include "gpuff_kernels_constants.cuh"

// 2. Transport kernels (puff movement)
#include "gpuff_kernels_transport.cuh"

// 3. Deposition and decay (material removal)
#include "gpuff_kernels_deposition.cuh"

// 4. Dispersion calculations (sigma parameters)
#include "gpuff_kernels_dispersion.cuh"

// 5. Concentration calculations (grid/receptor)
#include "gpuff_kernels_concentration.cuh"

// 6. Evacuation simulations (evacuee movement)
#include "gpuff_kernels_evacuation.cuh"

// 7. Exposure calculations (dose assessment)
#include "gpuff_kernels_exposure.cuh"

// 8. Utility functions (debug, testing)
#include "gpuff_kernels_utility.cuh"

// ====================================================================================
// CPU Function Declarations (from original file)
// ====================================================================================

namespace Gpuff {
    /**
     * CPU version of puff flag update
     * Used for validation and debugging
     */
    void update_puff_flags2_cpu(float currentTime, int nop);

    /**
     * CPU version of RCAP puff movement
     * Includes deposition and decay calculations
     */
    void move_puffs_by_wind_RCAP2_cpu(
        int EP_endRing, std::vector<NuclideData> ND, float* radius,
        int numRad, int numTheta, int nop);

    /**
     * CPU version of exposure calculation
     * Computes inhalation and cloudshine doses
     */
    void ComputeExposureHmix_cpu(
        std::vector<Evacuee> evacuees,
        ProtectionFactors PF,
        int numSims,
        int totalEvacueesPerSim,
        int totalPuffsPerSim);
}

/**
 * CPU version of evacuation calculation
 * Updates evacuee positions based on evacuation plan
 */
void evacuation_calculation_cpu(
    EvacuationDirections& ED, std::vector<Evacuee>& evacuee,
    float* radius, int numRad, int numTheta, int nop,
    int evaEndRing, int EP_endRing);

// ====================================================================================
// Global Device Variables (from original file)
// ====================================================================================

// Device constant memory declarations
__constant__ float d_dt;
__constant__ int d_nop;
__constant__ int d_time_end;
__constant__ int dimX;
__constant__ int dimY;
__constant__ int dimZ_pres;
__constant__ int dimZ_etas;
__constant__ int d_isPG;
__constant__ int d_isRural;
__constant__ float d_wc1;
__constant__ float d_wc2;
__constant__ float invPI;

// Simulation control constants
__constant__ int d_numSims;
__constant__ int d_totalevacuees_per_Sim;
__constant__ int d_totalpuff_per_Sim;

// Device global memory pointers
__device__ float* d_etas_hgt_uv;
__device__ float* d_etas_hgt_w;

// ====================================================================================
// Module Information
// ====================================================================================
//
// File Organization:
//   - gpuff_kernels_constants.cuh: ~430 lines - Constants, atomic ops, dispersion formulas
//   - gpuff_kernels_transport.cuh: ~400 lines - Puff movement and advection
//   - gpuff_kernels_deposition.cuh: ~440 lines - Deposition, decay, ground tracking
//   - gpuff_kernels_dispersion.cuh: ~280 lines - Dispersion parameter updates
//   - gpuff_kernels_concentration.cuh: ~240 lines - Concentration calculations
//   - gpuff_kernels_evacuation.cuh: ~230 lines - Evacuation simulations
//   - gpuff_kernels_exposure.cuh: ~600 lines - Exposure and dose calculations
//   - gpuff_kernels_utility.cuh: ~350 lines - Utility and debug functions
//
// Total: ~3000 lines (original: 3454 lines)
// Average per file: ~375 lines (target: 600-800 lines)
//
// ====================================================================================

#endif // GPUFF_KERNELS_CUH