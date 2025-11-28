// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel Implementation - Main Include File
// ====================================================================================
//
// File: gpuff_kernels.cuh
// Purpose: Main include file that aggregates all kernel implementations
//
// This file serves as the central include point for all GPUFF CUDA kernels.
// The implementation has been split into logical modules for better maintainability:
//
//   1. gpuff_kernels_dispersion.cuh - Atmospheric dispersion formulas
//   2. gpuff_kernels_puff.cuh - Puff transport and deposition
//   3. gpuff_kernels_evacuation.cuh - Evacuation simulation
//   4. gpuff_kernels_dose.cuh - Dose and exposure calculations
//   5. gpuff_kernels_cloudshine.cuh - Cloudshine external dose calculations
//
// ====================================================================================

#ifndef GPUFF_KERNELS_CUH
#define GPUFF_KERNELS_CUH

// Include all kernel implementation modules
#include "gpuff_kernels_dispersion.cuh"
#include "gpuff_kernels_puff.cuh"
#include "gpuff_kernels_evacuation.cuh"
#include "gpuff_kernels_dose.cuh"
#include "gpuff_kernels_cloudshine.cuh"

#endif // GPUFF_KERNELS_CUH