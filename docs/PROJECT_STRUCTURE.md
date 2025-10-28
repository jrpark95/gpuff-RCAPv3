# GPUFF-RCAPv3 Project Structure and Dependencies

## Overview

GPUFF-RCAPv3 is a CUDA-accelerated Gaussian Puff atmospheric dispersion model integrated with radiological consequence assessment capabilities. This document describes the project organization, file dependencies, and build requirements.

**Last Updated:** 2025-10-27

---

## Table of Contents

1. [Project Organization](#project-organization)
2. [File Dependencies](#file-dependencies)
3. [Build System](#build-system)
4. [Runtime Dependencies](#runtime-dependencies)
5. [Input/Output Files](#inputoutput-files)
6. [Data Flow](#data-flow)

---

## Project Organization

### Directory Structure

```
X:\code\gpuffv4\gpuff-RCAPv3\
│
├── Source Files
│   ├── main.cu                    # Main program entry point
│   ├── gpuff.cuh                  # Main class definition and global variables
│   ├── gpuff_struct.cuh           # Data structure definitions
│   ├── gpuff_kernels.cuh          # CUDA kernel implementations
│   ├── gpuff_kernels.h            # CUDA kernel declarations
│   ├── gpuff_init.cuh             # Initialization and file parsing
│   ├── gpuff_func.cuh             # Core simulation functions
│   ├── gpuff_mdata.cuh            # Meteorological data processing
│   └── gpuff_plot.cuh             # Output and visualization functions
│
├── Build System
│   ├── build.bat                  # Build script (Windows)
│   └── gpuff-RCAPv2.vcxproj       # Visual Studio project file
│
├── Documentation
│   ├── README.md                  # User guide and overview
│   ├── CODING_STANDARDS.md        # Coding conventions
│   └── PROJECT_STRUCTURE.md       # This file
│
├── Input Data
│   └── input\RCAPdata\
│       ├── Test1.inp              # Main scenario configuration
│       ├── Test2.inp - Test5.inp  # Additional scenarios
│       ├── MACCS60.NDL            # Nuclide decay library (80 nuclides)
│       ├── MACCS_DCF_New2.LIB     # Dose conversion factors
│       └── METEO.inp              # Meteorological time series
│
├── Output Directories
│   ├── output\                    # General output files
│   ├── evac\                      # Evacuee tracking data
│   ├── puffs\                     # Puff position and concentration
│   ├── plants\                    # Plant-specific output
│   └── receptors\                 # Receptor grid concentrations
│
└── Build Artifacts
    ├── gpuff.exe                  # Main executable
    ├── gpuff.lib                  # Import library
    └── gpuff.exp                  # Export file
```

---

## File Dependencies

### Dependency Graph

```
main.cu
  └── gpuff.cuh
       ├── <system headers>
       │    ├── vector, iostream, iomanip, fstream, sstream
       │    ├── algorithm, string, chrono, cstdio, cstdlib
       │    ├── unordered_map, map, math.h, limits, float.h
       │    └── cuda_runtime.h
       │
       └── gpuff_struct.cuh
            ├── Constants (MAX_NUCLIDES, MAX_ORGANS, EARTH_RADIUS, etc.)
            └── Structures
                 ├── SimulationControl
                 ├── RadioNuclideTransport
                 ├── NuclideData
                 ├── WeatherSamplingData
                 ├── EvacuationData
                 ├── EvacuationDirections
                 ├── SiteData
                 ├── ProtectionFactors
                 ├── HealthEffect
                 ├── Evacuee
                 ├── Puffcenter_RCAP
                 └── RCAP_METDATA

gpuff.cuh includes (implicitly via class implementation):
  ├── gpuff_init.cuh         # Parsing and initialization functions
  ├── gpuff_func.cuh         # Core simulation methods
  ├── gpuff_mdata.cuh        # Meteorological data handling
  ├── gpuff_plot.cuh         # Output generation
  └── gpuff_kernels.cuh      # CUDA kernel implementations
       └── gpuff_kernels.h   # Kernel function declarations
```

### Compilation Order

The NVCC compiler resolves dependencies automatically based on `#include` directives:

1. **First:** `gpuff_struct.cuh` (data structures, constants)
2. **Second:** `gpuff.cuh` (class definition, global variables)
3. **Third:** Implementation files (init, func, mdata, plot, kernels)
4. **Finally:** `main.cu` (program entry point)

**Note:** With `-rdc=true` (relocatable device code), device functions can call across compilation units.

---

## Build System

### Build Requirements

#### Hardware Requirements
- **GPU:** NVIDIA GPU with CUDA Compute Capability 5.0 or higher
  - Recommended: Compute Capability 7.0+ (Volta architecture or newer)
  - Minimum VRAM: 2 GB (4+ GB recommended for large simulations)

#### Software Requirements
- **Operating System:** Windows 10/11 64-bit
- **CUDA Toolkit:** Version 12.2 or later
  - Download: https://developer.nvidia.com/cuda-downloads
  - Components needed: NVCC compiler, CUDA runtime libraries
- **Visual Studio:** 2022 (Community, Professional, or Enterprise)
  - Required workload: "Desktop development with C++"
  - Platform toolset: v143
- **GPU Driver:** NVIDIA driver 525.60.11 or later (for CUDA 12.2)

### Build Process

#### Command-Line Build

```batch
# Navigate to project directory
cd X:\code\gpuffv4\gpuff-RCAPv3

# Run build script
build.bat

# Execute program
gpuff.exe
```

The `build.bat` script performs:
1. Initializes Visual Studio 2022 x64 environment
2. Validates prerequisites
3. Compiles CUDA source with NVCC
4. Links executable
5. Reports build status

#### Visual Studio Build

1. Open `gpuff-RCAPv2.vcxproj` in Visual Studio 2022
2. Select configuration (Debug or Release)
3. Build → Build Solution (Ctrl+Shift+B)
4. Debug → Start Without Debugging (Ctrl+F5)

### Compiler Flags

Current configuration:
```
nvcc main.cu -o gpuff.exe -allow-unsupported-compiler -rdc=true
```

**Flag Descriptions:**
- `-o gpuff.exe` : Specifies output executable name
- `-allow-unsupported-compiler` : Permits newer MSVC versions not officially supported
- `-rdc=true` : Enables relocatable device code (separate compilation)

**Optional Flags for Development:**
```
-G                    # Enable device debugging
-lineinfo             # Generate line number info for profiling
-Xcompiler "/W3"      # Warning level 3
```

**Optional Flags for Production:**
```
-O3                   # Maximum optimization
-use_fast_math        # Use fast math (slightly less precise)
-arch=sm_75           # Target specific GPU architecture
```

### Build Output

Successful build generates:
- `gpuff.exe` : Main executable (20-40 MB typical)
- `gpuff.lib` : Import library for linking
- `gpuff.exp` : Export file (if applicable)

---

## Runtime Dependencies

### GPU Runtime Requirements

- **CUDA Runtime:** Dynamically linked (cudart64_12.dll or similar)
- **GPU Memory:** Varies with simulation size
  - Base: ~100 MB (nuclide data, constants)
  - Per puff: ~3 KB (80 nuclides × 4 bytes + metadata)
  - Per evacuee: ~1 KB (dose tracking, position)
  - Meteorological data: ~50-500 MB (depends on grid resolution)

**Memory Estimation:**
```
Total GPU Memory ≈ 100 MB base
                 + (nPuffs × 3 KB)
                 + (nEvacuees × 1 KB)
                 + (meteorological grid size)
```

Example: 10,000 puffs + 50,000 evacuees ≈ 180 MB

### Input File Dependencies

#### Required Files

All files must be present in `.\input\RCAPdata\`:

1. **Test1.inp** (or Test.inp)
   - Main simulation configuration
   - Scenario parameters
   - Grid definition
   - Release characteristics

2. **MACCS60.NDL**
   - Nuclide decay library
   - 80 radionuclides (Kr-85m through Cm-244)
   - Decay chains and branching fractions
   - Size: ~50 KB

3. **MACCS_DCF_New2.LIB**
   - Dose conversion factors
   - 80 nuclides × 20 organs × 5 pathways
   - Size: ~100 KB

4. **METEO.inp**
   - Time-series meteorological data
   - Wind speed, direction, stability class
   - Precipitation data
   - Size: Varies (typically 10-500 KB)

#### Optional Files

- **Test2.inp - Test5.inp** : Additional scenarios for multi-run simulations

---

## Input/Output Files

### Input File Formats

#### Test.inp Structure

```
SC10: <Simulation Title>
SC20: <Plant Name> <Power Output [MWth]> <Plant Type>
SC30: <Number of Radii> <Number of Azimuthal Sectors>
SC31: <Radius 1> <Radius 2> ... <Radius N> [km]
SC40: <Weather File> <Nuclide File> <DCF File>

RT100: <Number of Input Files>
RT200: <Number of Puffs>
RT210: <Release Time> <Height> <Heat Rate> [for each puff]
RT220: <Particle Size Distribution> [10 bins]

RT310: <Weather Sampling Method>
RT350: <Wind Speed> <Stability Class> <Precipitation> [if manual mode]

EP200: <Alert Time> [s]
EP210: <Evacuation Start Ring> <End Ring>
EP220: <Shelter Wait Time> [s]
EP230: <Shelter Duration> [s]
EP240: <Number of Speed Periods>
EP241: <Speed 1> <Duration 1> [m/s] [s]
EP242: <Speed 2> <Duration 2>
...

SD50: <Surface Roughness> [m]
SD150: <Number of Population Rings>
SD151: <Inner Radius> <Outer Radius> <Population> [for each ring]

PF100: <Shielding Factor (building)>
PF200: <Shielding Factor (vehicle)>
```

#### MACCS60.NDL Format

```
<Nuclide Name> <ID> <Half-life [s]> <Atomic Weight [g/mol]>
<Chemical Group> <Dry Deposition Flag> <Wet Deposition Flag>
<Core Inventory [Ci/MWth]>
<Decay Count> <Daughter 1 ID> <Branching 1> <Daughter 2 ID> <Branching 2>
```

Repeat for all 80 nuclides.

#### MACCS_DCF_New2.LIB Format

```
<Nuclide Name>
<Organ 1 Name>
<Cloudshine> <Groundshine> <Inhalation> <Ingestion> <Resuspension>
<Organ 2 Name>
<Cloudshine> <Groundshine> <Inhalation> <Ingestion> <Resuspension>
...
[Repeat for 20 organs]
```

Repeat for all 80 nuclides.

#### METEO.inp Format

```
<Date> <Time> <Wind Direction [deg]> <Wind Speed [m/s]> <Stability Class> <Precipitation [mm/h]>
```

One line per timestep (typically hourly data).

### Output File Formats

#### Puff Output (puffs\puff_XXXX.bin)

Binary format:
```
[int32] timestep
[int32] nPuffs
[struct Puff] × nPuffs
  - float x, y, z          [m] (position)
  - float conc[80]         [Bq] (nuclide concentrations)
  - float sigma_h, sigma_z [m] (dispersion coefficients)
  - int flag               (active/inactive)
```

#### Evacuee Output (evac\evac_XXXX.bin)

Binary format:
```
[int32] timestep
[int32] nEvacuees
[struct Evacuee] × nEvacuees
  - float x, y             [m] (position)
  - int population         (number of people)
  - float dose_inh         [Sv] (inhalation dose)
  - float dose_cloud       [Sv] (cloudshine dose)
  - float organ_dose[20]   [Sv] (organ-specific doses)
```

#### Plant Output (plants\plant_X_XXXX.bin)

Binary format for each scenario (X = scenario index):
```
[int32] timestep
[float] concentrations[80]  [Bq/m³] at specific location
```

---

## Data Flow

### Initialization Phase

```
main() starts
    │
    ├─> Read MACCS_DCF_New2.LIB
    │    └─> Parse dose conversion factors → NuclideData[]
    │
    ├─> Read MACCS60.NDL
    │    └─> Parse decay chains, half-lives → NuclideData[]
    │
    ├─> Flatten exposure data for GPU
    │    └─> exposure_data_all[80 × 20 × 5]
    │
    ├─> Copy nuclide data to GPU
    │    └─> cudaMemcpy(..., cudaMemcpyHostToDevice)
    │
    ├─> Read Test.inp (and Test1-5.inp if multi-run)
    │    ├─> SimulationControl (grid, timing)
    │    ├─> RadioNuclideTransport (release data)
    │    ├─> EvacuationData (evacuation plan)
    │    ├─> SiteData (population, terrain)
    │    └─> ProtectionFactors (shielding)
    │
    ├─> Read METEO.inp
    │    └─> Time-series wind, stability, precipitation
    │
    ├─> gpuff.initializePuffs()
    │    └─> Create puff objects with release schedule
    │
    ├─> gpuff.initializeEvacuees()
    │    └─> Populate evacuees on polar grid
    │
    └─> Copy all data to GPU
         ├─> d_puffs
         ├─> d_evacuees
         ├─> d_ND (nuclide data)
         └─> d_ground_deposit
```

### Simulation Loop

```
for t = 0 to time_end by dt:
    │
    ├─> update_puff_flags() kernel
    │    └─> Check release times, activate puffs
    │
    ├─> move_puffs_by_wind() kernel
    │    ├─> Interpolate wind field at puff location
    │    ├─> Advect puff (x += u*dt, y += v*dt)
    │    └─> Grow dispersion coefficients (σ_h, σ_z)
    │
    ├─> radioactive_decay() kernel
    │    └─> Apply decay: Q(t) = Q(0) * exp(-λt)
    │         └─> Handle decay chains (parent → daughter)
    │
    ├─> deposition() kernel
    │    ├─> Dry deposition: dQ/dt = -v_d * C_ground
    │    └─> Wet deposition: dQ/dt = -Λ * Q  (if rain)
    │
    ├─> accumulate_ground_deposit() kernel
    │    └─> Add deposited mass to ground grid
    │
    ├─> move_evacuees() kernel
    │    └─> Update evacuee positions based on speed profile
    │
    ├─> ComputeExposureHmix() kernel
    │    ├─> For each evacuee:
    │    │    └─> For each puff:
    │    │         ├─> Calculate concentration at evacuee location
    │    │         │    (Gaussian puff equation)
    │    │         ├─> Accumulate inhalation dose
    │    │         │    D_inh += C * BR * DCF_inh * dt
    │    │         └─> Accumulate cloudshine dose
    │    │              D_cloud += C * DCF_cloud * dt
    │    │
    │    └─> Accumulate organ-specific doses
    │
    ├─> reduce_organDose() kernel
    │    └─> Sum doses across all nuclides for each organ
    │
    └─> Output results (if t % freq_output == 0)
         ├─> Copy puffs from device to host
         ├─> Copy evacuees from device to host
         ├─> Write binary output files
         └─> Write text summaries (optional)
```

### Finalization Phase

```
End of simulation
    │
    ├─> Copy final results to host
    │
    ├─> Write final output files
    │
    ├─> gpuff.free_puffs_RCAP_device_memory()
    │    └─> cudaFree(d_puffs, d_evacuees, d_ND, ...)
    │
    └─> Print execution time statistics
```

---

## Key Data Structures

### Global Constants

| Constant | Value | Unit | Description |
|----------|-------|------|-------------|
| MAX_NUCLIDES | 80 | - | Maximum number of radionuclides |
| MAX_ORGANS | 20 | - | Maximum number of organs for dose calculation |
| DATA_FIELDS | 5 | - | Exposure pathways (cloudshine, groundshine, inhalation, ingestion, resuspension) |
| EARTH_RADIUS | 6,371,000 | m | Mean Earth radius |
| PI | 3.141592 | - | Mathematical constant π |

### Memory Usage Summary

| Data Structure | Count | Size per Item | Total Size |
|----------------|-------|---------------|------------|
| NuclideData | 80 | ~1.2 KB | ~100 KB |
| Puffcenter_RCAP | Variable | ~3 KB | nPuffs × 3 KB |
| Evacuee | Variable | ~1 KB | nEvacuees × 1 KB |
| Ground deposit grid | nRings × nTheta | 80 floats | nRings × nTheta × 320 bytes |

---

## Performance Considerations

### GPU Utilization

- **Puff kernels:** 1 thread per puff (1D grid)
- **Evacuee kernels:** 1 thread per evacuee (1D grid)
- **Dose calculation:** nEvacuees × nPuffs thread pairs (2D grid)

### Optimization Strategies

1. **Memory Coalescing:** Align data structures to 128-byte boundaries
2. **Constant Memory:** Use `__constant__` for simulation parameters
3. **Shared Memory:** Cache frequently accessed data within thread blocks
4. **Atomic Operations:** Minimize atomics; use reduction patterns where possible
5. **Occupancy:** Balance register usage vs. thread count

### Scalability

| Parameter | Small | Medium | Large |
|-----------|-------|--------|-------|
| Puffs | 1,000 | 10,000 | 100,000 |
| Evacuees | 10,000 | 50,000 | 500,000 |
| GPU Memory | ~500 MB | ~2 GB | ~10 GB |
| Execution Time | Minutes | Tens of minutes | Hours |

---

## Troubleshooting

### Build Issues

**Problem:** CUDA Toolkit not found
- **Solution:** Add CUDA `bin` directory to system PATH
  - Default: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin`

**Problem:** Visual Studio not found
- **Solution:** Update `VS_PATH` in `build.bat` to match your installation

**Problem:** Compilation errors about unsupported compiler
- **Solution:** Use `-allow-unsupported-compiler` flag (already in build.bat)

### Runtime Issues

**Problem:** Out of GPU memory
- **Solution:** Reduce number of puffs or evacuees, or use a GPU with more VRAM

**Problem:** Input file not found
- **Solution:** Verify all required files are in `.\input\RCAPdata\` directory

**Problem:** Slow execution
- **Solution:**
  - Reduce output frequency (`freq_output`)
  - Use `-O3 -use_fast_math` compiler flags
  - Target specific GPU architecture with `-arch=sm_XX`

---

## Contact and Support

For technical questions regarding project structure, build system, or dependencies:
- Review documentation in this directory
- Check CUDA Toolkit documentation
- Consult MACCS manual (NUREG/CR-6613)

---

**Last updated:** 2025-10-27
