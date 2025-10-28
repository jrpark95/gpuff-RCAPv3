# GPUFF-RCAPv3 Quick Start Guide

## Overview

This guide provides step-by-step instructions to build and run GPUFF-RCAPv3 on Windows systems.

**Target Audience:** Developers and researchers who want to quickly get started

---

## Prerequisites

Before building, ensure you have:

### Required Software

1. **NVIDIA CUDA Toolkit 12.2 or later**
   - Download: https://developer.nvidia.com/cuda-downloads
   - Installation: Follow NVIDIA installer instructions
   - Verify: Open command prompt and run `nvcc --version`

2. **Visual Studio 2022**
   - Download: https://visualstudio.microsoft.com/downloads/
   - Edition: Community (free), Professional, or Enterprise
   - Required Workload: "Desktop development with C++"
   - Verify: Check installation at `C:\Program Files\Microsoft Visual Studio\2022\`

3. **NVIDIA GPU**
   - Compute Capability: 5.0 or higher
   - Driver: Version 525.60.11 or later
   - Verify: Run `nvidia-smi` in command prompt

### Required Input Files

Ensure these files exist in `.\input\RCAPdata\`:

- `Test1.inp` - Main simulation configuration
- `MACCS60.NDL` - Nuclide decay library
- `MACCS_DCF_New2.LIB` - Dose conversion factors
- `METEO.inp` - Meteorological data

---

## Build Instructions

### Method 1: Command Line (Recommended)

```batch
# Step 1: Open Command Prompt
# Press Win+R, type "cmd", press Enter

# Step 2: Navigate to project directory
cd X:\code\gpuffv4\gpuff-RCAPv3

# Step 3: Run build script
build.bat

# Step 4: Wait for compilation (typically 1-3 minutes)
# You should see:
#   [1/3] Initializing Visual Studio environment...
#   [2/3] Compiling CUDA source files...
#   [3/3] Build complete!
```

**Expected Output:**
```
============================================================================
GPUFF-RCAPv3 Build System
============================================================================

[1/3] Initializing Visual Studio 2022 x64 environment...
Visual Studio environment initialized successfully.

[2/3] Compiling CUDA source files...
Compiling main.cu with nvcc...
Compilation successful.

[3/3] Build complete!
============================================================================
Output executable: gpuff.exe
============================================================================
```

### Method 2: Visual Studio IDE

```
1. Open gpuff-RCAPv2.vcxproj in Visual Studio 2022
2. Select Configuration: Release (or Debug for debugging)
3. Select Platform: x64
4. Press Ctrl+Shift+B (or Build → Build Solution)
5. Wait for compilation
6. Executable will be in x64\Release\ (or x64\Debug\)
```

---

## Running the Simulation

### Basic Execution

```batch
# After successful build:
gpuff.exe

# The program will:
# 1. Load input files from .\input\RCAPdata\
# 2. Initialize puffs and evacuees
# 3. Run time-stepping simulation
# 4. Output results to .\output\, .\evac\, .\puffs\, .\plants\
```

### Expected Console Output

```
Size of NuclideData: 1024
File 1 of 1 = .\input\RCAPdata\Test1.inp

nop = 1000
totalevacuees_per_Sim = 50000
totalpuff_per_Sim = 1000

[Simulation running...]

Total execution time: 1234.56 ms
```

---

## Output Files

### Output Directory Structure

```
gpuff-RCAPv3\
├── output\          # General output files
├── evac\            # Evacuee tracking data (binary)
│   └── evac_XXXX.bin
├── puffs\           # Puff positions and concentrations (binary)
│   └── puff_XXXX.bin
├── plants\          # Plant-specific output (binary)
│   └── plant_X_XXXX.bin
└── receptors\       # Receptor grid data
```

### Reading Output Files

Output files are in binary format. To read:

1. **Puff files:** Use VTK-compatible readers (ParaView, VisIt)
2. **Evacuee files:** Custom binary reader (see PROJECT_STRUCTURE.md)
3. **Text output:** Console output shows key statistics

---

## Configuration

### Adjusting Simulation Parameters

Edit `.\input\RCAPdata\Test1.inp`:

```
SC10: <Simulation Title>
SC20: <Plant> <Power [MWth]> <Type>
SC30: <nRadii> <nTheta>         # Grid resolution
...
```

### Debug Flags

Edit `gpuff.cuh` to enable debug output:

```cpp
#define CHECK_SC 1       // Print simulation control
#define CHECK_RT 1       // Print release transport
#define CHECK_NDL 1      // Print nuclide library
```

Rebuild after changes:
```batch
build.bat
```

---

## Troubleshooting

### Build Failures

**Problem:** "nvcc is not recognized"
```
Solution:
1. Verify CUDA installation
2. Add to PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin
3. Restart command prompt
```

**Problem:** "Visual Studio not found"
```
Solution:
1. Check installation path in build.bat (line 45)
2. Update VS_PATH variable if installation is elsewhere
3. Example: set VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\...
```

**Problem:** "Compilation failed"
```
Solution:
1. Check error messages in console
2. Verify all .cuh files are present
3. Ensure no syntax errors in source files
4. Try clean build (delete gpuff.exe, gpuff.lib, gpuff.exp first)
```

### Runtime Failures

**Problem:** "Input file not found"
```
Solution:
1. Verify files exist in .\input\RCAPdata\
2. Check file names (case-sensitive on some systems)
3. Ensure files are not corrupted
```

**Problem:** "Out of memory"
```
Solution:
1. Check GPU VRAM: nvidia-smi
2. Reduce number of puffs in Test1.inp
3. Reduce number of evacuees
4. Use smaller grid resolution
```

**Problem:** "Slow execution"
```
Solution:
1. Reduce output frequency in Test1.inp
2. Rebuild with optimization: nvcc -O3 -use_fast_math
3. Target specific GPU: nvcc -arch=sm_75 (adjust for your GPU)
```

### Getting GPU Architecture

```batch
# Check your GPU compute capability:
nvidia-smi --query-gpu=compute_cap --format=csv

# Example output:
# compute_cap
# 7.5

# Use this for -arch flag:
# sm_75 for compute capability 7.5
```

---

## Performance Tips

### For Development (Fast Compilation)

```batch
# Use default build.bat (no changes needed)
build.bat
```

### For Production (Maximum Performance)

Edit `build.bat` line 108:
```batch
# Change from:
nvcc main.cu -o gpuff.exe -allow-unsupported-compiler -rdc=true

# To:
nvcc main.cu -o gpuff.exe -allow-unsupported-compiler -rdc=true -O3 -use_fast_math -arch=sm_75
```

Replace `sm_75` with your GPU's architecture (see above).

---

## Next Steps

### Understanding the Code

1. Read `README.md` - Overview of physics models
2. Read `PROJECT_STRUCTURE.md` - File organization and dependencies
3. Read `CODING_STANDARDS.md` - Coding conventions

### Modifying the Code

1. Follow patterns in CODING_STANDARDS.md
2. Add function documentation for new functions
3. Test thoroughly before committing
4. Keep comments in English

### Visualizing Results

1. Install ParaView: https://www.paraview.org/download/
2. Open .vtk files from .\puffs\ or .\evac\
3. Apply filters for visualization
4. Create animations of puff evolution

---

## Common Workflows

### Running Multiple Scenarios

```batch
# Edit MU10 section in Test1.inp to list multiple input files
MU10: 5 scenario1.inp scenario2.inp scenario3.inp scenario4.inp scenario5.inp

# Run once - all scenarios will be processed
gpuff.exe
```

### Changing Meteorological Data

```batch
# Edit METEO.inp with new wind data
# Format: Date Time WindDir[deg] WindSpeed[m/s] StabilityClass Precip[mm/h]
2024-01-01 00:00:00 180 5.0 4 0.0
2024-01-01 01:00:00 185 5.5 4 0.0
...

# Rebuild not needed - just rerun
gpuff.exe
```

### Clean Build

```batch
# Delete build artifacts
del gpuff.exe gpuff.lib gpuff.exp

# Rebuild from scratch
build.bat
```

---

## Support and Documentation

### Documentation Files

| File | Purpose |
|------|---------|
| QUICK_START.md | This file - getting started |
| README.md | User guide and physics overview |
| PROJECT_STRUCTURE.md | Technical architecture |
| CODING_STANDARDS.md | Development guidelines |
| MODERNIZATION_SUMMARY.md | Recent changes |

### Getting Help

1. Check documentation files above
2. Review error messages carefully
3. Verify prerequisites are met
4. Check input file formats
5. Consult MACCS manual (NUREG/CR-6613) for physics questions

---

## Example Session

Complete workflow from build to visualization:

```batch
# 1. Build the project
cd X:\code\gpuffv4\gpuff-RCAPv3
build.bat

# 2. Verify input files exist
dir input\RCAPdata\Test1.inp
dir input\RCAPdata\MACCS60.NDL
dir input\RCAPdata\METEO.inp

# 3. Run simulation
gpuff.exe

# 4. Check output
dir output
dir evac
dir puffs

# 5. Visualize (requires ParaView)
# Open ParaView → File → Open → puffs\puff_0000.vtk
# Apply filters and explore results
```

---

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1050 (CC 5.0) | RTX 3060 or better (CC 7.5+) |
| VRAM | 2 GB | 8 GB or more |
| System RAM | 8 GB | 16 GB or more |
| Disk Space | 1 GB | 10 GB (with outputs) |
| OS | Windows 10 64-bit | Windows 11 64-bit |
| CUDA | 12.2 | 12.6 or latest |
| Visual Studio | 2022 Community | 2022 Professional/Enterprise |

---

## Frequently Asked Questions

**Q: How long does compilation take?**
A: Typically 1-3 minutes on modern systems.

**Q: How long does a simulation take?**
A: Depends on parameters. Typical: 1-30 minutes. Large: hours.

**Q: Can I run without a GPU?**
A: No, CUDA GPU is required. CPU-only version not available.

**Q: Can I use Visual Studio 2019?**
A: Possibly, but may require changes. VS 2022 is officially supported.

**Q: What GPU architectures are supported?**
A: Compute Capability 5.0 or higher (Maxwell architecture and newer).

**Q: How do I cite this software?**
A: See README.md for citation information.

---

**Last Updated:** 2025-10-27
**Version:** 1.0
**For:** GPUFF-RCAPv3

---

Ready to get started? Run `build.bat` now!
