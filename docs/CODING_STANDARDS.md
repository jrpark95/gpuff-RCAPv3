# GPUFF-RCAPv3 Coding Standards

## Document Purpose

This document defines the coding standards and modernization guidelines applied to the GPUFF-RCAPv3 codebase. All code modifications should follow these standards to maintain consistency and readability.

**Last Updated:** 2025-10-27
**Applies To:** All source files (.cu, .cuh, .h)

---

## Table of Contents

1. [General Principles](#general-principles)
2. [File Organization](#file-organization)
3. [Naming Conventions](#naming-conventions)
4. [Code Formatting](#code-formatting)
5. [Documentation Standards](#documentation-standards)
6. [CUDA-Specific Guidelines](#cuda-specific-guidelines)
7. [Comments and Documentation](#comments-and-documentation)
8. [Dead Code Management](#dead-code-management)

---

## General Principles

### Core Rules

1. **DO NOT modify calculation logic** - Preserve all physics computations exactly as implemented
2. **English comments only** - All comments must be in English (no Korean)
3. **Maintain build process** - Do not break existing compilation or linking
4. **Improve clarity** - Focus on readability and maintainability
5. **Document thoroughly** - Explain purpose, parameters, and physics models

### Code Quality Goals

- **Readability:** Code should be self-documenting with clear variable names
- **Maintainability:** Future developers should understand the code easily
- **Consistency:** Apply patterns uniformly across the entire codebase
- **Performance:** Preserve GPU optimization patterns (no degradation)

---

## File Organization

### File Header Format

All files must begin with a standardized header block:

```cpp
// ============================================================================
// <Module Name>
// ============================================================================
// <Brief description of file purpose>
//
// Main Responsibilities:
// - <Responsibility 1>
// - <Responsibility 2>
// - <Responsibility 3>
//
// <Additional context or physics description>
// ============================================================================
```

### Section Dividers

Use consistent section dividers to organize code:

```cpp
// ============================================================================
// Major Section Name
// ============================================================================

// ----------------------------------------------------------------------------
// Subsection Name
// ----------------------------------------------------------------------------
```

### Include Order

1. Main header (if .cu/.cpp file)
2. Standard library headers (alphabetical)
3. Third-party library headers
4. Project headers (alphabetical)

Example:
```cpp
#include "gpuff.cuh"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "gpuff_init.cuh"
#include "gpuff_kernels.cuh"
#include "gpuff_struct.cuh"
```

---

## Naming Conventions

### Constants

- **Preprocessor Defines:** `UPPER_SNAKE_CASE`
  - Example: `MAX_NUCLIDES`, `EARTH_RADIUS`, `DATA_FIELDS`

- **constexpr Values:** `UPPER_SNAKE_CASE` or `PascalCase`
  - Example: `GRID_SPACING`, `MIN_PUFF_HEIGHT`

### Variables

- **Local Variables:** `camelCase`
  - Example: `currentTime`, `puffIndex`, `windSpeed`

- **Struct/Class Members:** `camelCase`
  - Example: `half_life`, `atomic_weight`, `decay_count`

- **Global Variables:** `camelCase` (avoid when possible)
  - Example: `time_end`, `dt`, `freq_output`

- **Device Pointers:** Prefix with `d_`
  - Example: `d_puffs`, `d_ND`, `d_evacuees`

### Functions

- **Regular Functions:** `camelCase` or `snake_case`
  - Example: `initializeNuclideData()`, `read_MACCS_DCF_New2()`

- **CUDA Kernels:** `snake_case` or `camelCase`
  - Example: `update_puff_flags_RCAP2`, `ComputeExposureHmix`

### Structures and Classes

- **Type Names:** `PascalCase`
  - Example: `NuclideData`, `SimulationControl`, `Puffcenter_RCAP`

- **Enum Values:** `UPPER_SNAKE_CASE`
  - Example: `STABILITY_CLASS_A`, `STABILITY_CLASS_D`

---

## Code Formatting

### Indentation and Spacing

- **Indentation:** 4 spaces (no tabs)
- **Line Length:** Maximum 100 characters (soft limit), 120 (hard limit)
- **Blank Lines:** One blank line between functions, two between major sections

### Braces

Use consistent brace style (K&R or Allman, but be consistent):

```cpp
// Preferred (K&R style)
if (condition) {
    doSomething();
}

// Also acceptable (Allman style) - be consistent within file
if (condition)
{
    doSomething();
}
```

### Whitespace

```cpp
// Around operators
float result = a + b * c;

// After commas
function(arg1, arg2, arg3);

// No space before semicolon
statement;

// Space after control keywords
if (condition) { }
for (int i = 0; i < n; i++) { }
while (running) { }
```

### Line Breaks for Long Statements

```cpp
// Break before operator for long expressions
float long_calculation =
    term1 * coefficient1 +
    term2 * coefficient2 +
    term3 * coefficient3;

// Break after comma for function calls
cudaMemcpy(d_exposure, exposure_data_all,
           sizeof(float) * MAX_NUCLIDES * MAX_ORGANS * DATA_FIELDS,
           cudaMemcpyHostToDevice);
```

---

## Documentation Standards

### Function Documentation

All functions should have documentation blocks:

```cpp
/**
 * Brief description of function purpose
 *
 * Detailed explanation if needed, including:
 * - Algorithm description
 * - Physics model reference
 * - Performance characteristics
 *
 * @param param1 Description of first parameter
 * @param param2 Description of second parameter
 * @return Description of return value
 *
 * @note Special considerations or warnings
 */
returnType functionName(Type1 param1, Type2 param2);
```

### Inline Comments

```cpp
// Single-line comment explaining next block of code
someCode();

// For complex logic, explain the "why" not just "what"
// Using Pasquill-Gifford model because rural dispersion is dominant
float sigma = calculateSigma(distance, stabilityClass);
```

### Physics Documentation

When implementing physics calculations, document:

1. **Model Name:** Pasquill-Gifford, Briggs-McElroy-Pooler, etc.
2. **Equations:** Reference or inline LaTeX-style notation
3. **Assumptions:** Atmospheric conditions, boundary conditions
4. **Units:** Explicitly state units for all physical quantities
5. **References:** Scientific papers or manuals

Example:
```cpp
// ============================================================================
// Gaussian Puff Concentration Model
// ============================================================================
// Calculates ground-level concentration from a Gaussian puff:
//
// C(x,y,z) = Q / ((2*PI)^1.5 * sigma_x * sigma_y * sigma_z)
//            * exp(-((x-x_p)^2/(2*sigma_x^2) +
//                    (y-y_p)^2/(2*sigma_y^2) +
//                    (z-z_p)^2/(2*sigma_z^2)))
//
// Where:
//   Q       = Total mass in puff [Bq]
//   sigma_x = Horizontal dispersion coefficient [m]
//   sigma_y = Lateral dispersion coefficient [m]
//   sigma_z = Vertical dispersion coefficient [m]
//   (x_p, y_p, z_p) = Puff center coordinates [m]
//
// Reference: Turner, D.B. (1994) "Workbook of Atmospheric Dispersion Estimates"
// ============================================================================
```

---

## CUDA-Specific Guidelines

### Kernel Launch Configuration

Document kernel launch parameters:

```cpp
// Launch configuration for puff update kernel
// - Grid: (nPuffs + 255) / 256 blocks
// - Block: 256 threads
// - Each thread processes one puff
dim3 gridDim((nPuffs + 255) / 256);
dim3 blockDim(256);
update_puffs_kernel<<<gridDim, blockDim>>>(d_puffs, nPuffs);
```

### Memory Transfer Documentation

```cpp
// Copy nuclide data from host to device
// Size: 80 nuclides * sizeof(NuclideData) ≈ 100KB
cudaMemcpy(d_ND, ND.data(),
           MAX_NUCLIDES * sizeof(NuclideData),
           cudaMemcpyHostToDevice);
```

### Device Function Annotations

```cpp
/**
 * Atomic minimum operation for floating-point values
 * Thread-safe operation for finding minimum across GPU threads
 * Uses Compare-And-Swap (CAS) pattern
 */
__device__ float atomicMinFloat(float* address, float val);

/**
 * Hybrid function callable from both host and device
 */
__host__ __device__ float calculateDistance(float x1, float y1, float x2, float y2);
```

### Constant Memory Usage

```cpp
// Device constant memory (cached, read-only)
__constant__ float d_time_end;      // Simulation end time [s]
__constant__ float d_dt;            // Time step [s]
__constant__ int d_nop;             // Number of puffs
```

---

## Comments and Documentation

### File-Level Comments

Every file must have:

1. **Purpose:** What the file does
2. **Contents:** Major structures/functions
3. **Dependencies:** External libraries or specific requirements
4. **Modification Log:** Keep track of significant changes (optional)

### Function-Level Comments

Required for all public functions and kernels:

1. **Brief:** One-line summary
2. **Detailed:** Algorithm explanation
3. **Parameters:** All inputs and outputs
4. **Returns:** What the function returns
5. **Side Effects:** Global state changes, file I/O, etc.
6. **Thread Safety:** For parallel code
7. **Performance:** Complexity, optimization notes

### Block-Level Comments

Use for logical blocks within functions:

```cpp
// Parse input file and extract scenario parameters
while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '!') {
        continue;  // Skip empty lines and comments
    }

    // Extract keyword from line
    std::istringstream iss(line);
    std::string keyword;
    iss >> keyword;

    // Process based on keyword type
    if (keyword == "SC10") {
        // Simulation control section
        parseSimulationControl(iss, SC);
    }
}
```

### Variable Documentation

For complex data structures:

```cpp
struct NuclideData {
    char name[MAX_STRING_LENGTH];        // Nuclide symbol (e.g., "Cs-137")
    int id;                              // Unique identifier (1-80)
    float half_life;                     // Decay half-life [seconds]
    float atomic_weight;                 // Atomic mass [g/mol]
    char chemical_group[MAX_STRING_LENGTH]; // Chemical classification
    int chemG_ID;                        // Chemical group ID (1=Xe, 2=I, etc.)
    float core_inventory;                // Core inventory [Ci/MWth]

    // Decay chain information
    int decay_count;                     // Number of decay products (0-2)
    int daughter[MAX_DNUC];              // Daughter nuclide IDs
    float branching_fraction[MAX_DNUC];  // Branching ratios (sum = 1.0)

    // Exposure data: 20 organs × 5 coefficients
    // [organ][0] = cloudshine DCF [Sv·m³/Bq·s]
    // [organ][1] = groundshine DCF [Sv·m²/Bq]
    // [organ][2] = inhalation DCF [Sv/Bq]
    // [organ][3] = ingestion DCF [Sv/Bq]
    // [organ][4] = resuspension DCF [Sv·m³/Bq·s]
    float exposure_data[MAX_ORGANS][DATA_FIELDS];
};
```

---

## Dead Code Management

### Removing Commented Code

**DO:** Remove old commented-out code if:
- It's been replaced by better implementation
- It's outdated or no longer relevant
- It clutters the codebase unnecessarily

**DON'T:** Remove commented code if:
- It documents alternative approaches
- It's needed for debugging or verification
- It represents future implementation options

### Marking TODO/FIXME/NOTE

Use standard comment markers:

```cpp
// TODO: Implement wet deposition for particle sizes > 10 microns
// FIXME: This approximation fails for stability class G in urban environments
// NOTE: Using simplified Briggs equations for z < 100m
// HACK: Temporary workaround for CUDA 12.x compiler bug
// OPTIMIZE: This loop could be vectorized for better performance
```

---

## Build System Standards

### Build Script Requirements

The build script (`build.bat`) must include:

1. **Header Block:** Purpose, requirements, usage
2. **Step Documentation:** Clear explanation of each build phase
3. **Error Checking:** Validation of prerequisites
4. **User Guidance:** Instructions for running and troubleshooting
5. **Developer Notes:** Optimization flags, architecture targeting

### Compiler Flags

Document all compiler flags:

```batch
REM -allow-unsupported-compiler : Permits newer Visual Studio versions
REM -rdc=true                   : Relocatable device code for separate compilation
REM -O3                         : Maximum optimization (production builds)
REM -G                          : Device debug symbols (development builds)
REM -lineinfo                   : Line number information for profiling
REM -arch=sm_XX                 : Target specific GPU architecture
```

---

## Verification Checklist

Before committing code, verify:

- [ ] All comments are in English
- [ ] No Korean characters in source files
- [ ] File headers present and complete
- [ ] Function documentation present for all public functions
- [ ] Constants follow `UPPER_SNAKE_CASE` convention
- [ ] No breaking changes to calculation logic
- [ ] Build script runs successfully
- [ ] Dead code removed or clearly marked
- [ ] Physics models documented with references
- [ ] GPU launch configurations documented
- [ ] Memory transfers have size comments

---

## Examples

### Good Code Example

```cpp
// ============================================================================
// Atmospheric Stability Classification
// ============================================================================

/**
 * Determines Pasquill-Gifford stability class from meteorological data
 *
 * Uses solar radiation and wind speed to classify atmospheric stability
 * following the Pasquill-Gifford scheme (classes A through F)
 *
 * @param windSpeed Wind speed at 10m height [m/s]
 * @param solarRadiation Incoming solar radiation [W/m²]
 * @param isDay True if daytime, false if nighttime
 * @return Stability class (0=A, 1=B, 2=C, 3=D, 4=E, 5=F)
 *
 * Reference: Pasquill, F. (1961) "The estimation of the dispersion of
 *            windborne material", Meteorological Magazine, 90, 33-49
 */
int calculateStabilityClass(float windSpeed, float solarRadiation, bool isDay) {
    // Nighttime conditions default to class E or F
    if (!isDay) {
        return (windSpeed < 2.0f) ? STABILITY_CLASS_F : STABILITY_CLASS_E;
    }

    // Daytime classification based on wind speed and insolation
    if (windSpeed < 2.0f) {
        return (solarRadiation > 600.0f) ? STABILITY_CLASS_A : STABILITY_CLASS_B;
    } else if (windSpeed < 3.0f) {
        return (solarRadiation > 600.0f) ? STABILITY_CLASS_B : STABILITY_CLASS_C;
    } else if (windSpeed < 5.0f) {
        return (solarRadiation > 300.0f) ? STABILITY_CLASS_C : STABILITY_CLASS_D;
    } else {
        return STABILITY_CLASS_D;  // Neutral conditions for high wind speeds
    }
}
```

### Bad Code Example (Avoid This)

```cpp
// 안정도 계산  (WRONG: Korean comment)
int calcSC(float ws,float sr,bool d){  // WRONG: unclear names, no documentation
float x=ws;  // WRONG: meaningless variable name
if(!d)return x<2.0f?5:4;  // WRONG: magic numbers, unreadable
// some old code  (WRONG: vague comment about dead code)
//if(ws<2&&sr>600)return 0;
if(x<2.0f)return sr>600.0f?0:1;  // WRONG: inconsistent formatting
else if(x<3.0f)return sr>600.0f?1:2;
else if(x<5.0f)return sr>300.0f?2:3;
else return 3;
}
```

---

## Modernization Progress

### Completed Files

The following files have been modernized according to these standards:

- `build.bat` - Build script with comprehensive documentation
- `main.cu` - Main entry point with structured comments
- `gpuff_kernels.cuh` - CUDA kernels with physics documentation
- `gpuff_init.cuh` - Initialization functions with data format specs
- `gpuff_func.cuh` - Core functions with clear documentation
- `gpuff_mdata.cuh` - Meteorological data processing

### Common Patterns Applied

1. **File Headers:** All files have standardized headers
2. **Section Dividers:** Consistent `===` and `---` dividers
3. **Function Documentation:** Doxygen-style comment blocks
4. **Inline Comments:** Explain "why" not just "what"
5. **Constants:** Named constants replace magic numbers
6. **Dead Code:** Removed or clearly marked for retention

---

## References

- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- MACCS Code Manual: NUREG/CR-6613
- Pasquill-Gifford Dispersion Model: Turner (1994)
- C++ Core Guidelines: https://isocpp.github.io/CppCoreGuidelines/

---

## Version History

- **v1.0** (2025-10-27): Initial coding standards document
  - Established naming conventions
  - Documented file organization standards
  - Defined documentation requirements
  - Created modernization checklist

---

**For questions or suggestions regarding these coding standards, please contact the development team.**
