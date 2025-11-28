@echo off
setlocal enabledelayedexpansion

echo ============================================================================
echo GPUFF-RCAPv3 Unified Build Script
echo Automatically detecting CUDA and Visual Studio environment...
echo ============================================================================
echo.

REM Initialize variables
set "NVCC_PATH="
set "CUDA_VERSION="
set "VS_COMPILER="
set "VS_VERSION="
set "EXTRA_FLAGS="

REM ============================================================================
REM STEP 1: Detect CUDA installation
REM ============================================================================
echo [1/3] Detecting CUDA installation...

REM Check for different CUDA versions (newest to oldest)
for %%v in (12.6 12.5 12.4 12.3 12.2 12.1 12.0 11.8 11.7 11.6 11.5 11.4 11.3 11.2 11.1 11.0) do (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%v\bin\nvcc.exe" (
        set "NVCC_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%v\bin\nvcc.exe"
        set "CUDA_VERSION=%%v"
        goto :cuda_found
    )
)

REM Also check if CUDA_PATH environment variable is set
if defined CUDA_PATH (
    if exist "%CUDA_PATH%\bin\nvcc.exe" (
        set "NVCC_PATH=%CUDA_PATH%\bin\nvcc.exe"
        for /f "tokens=3" %%i in ('"%NVCC_PATH%" --version ^| findstr "release"') do (
            set "CUDA_VERSION=%%i"
        )
        goto :cuda_found
    )
)

echo ERROR: CUDA not found!
echo Please install CUDA from: https://developer.nvidia.com/cuda-downloads
echo.
pause
exit /b 1

:cuda_found
echo [OK] Found CUDA %CUDA_VERSION% at: %NVCC_PATH%
echo.

REM ============================================================================
REM STEP 2: Detect Visual Studio installation
REM ============================================================================
echo [2/3] Detecting Visual Studio installation...

REM Check for VS2022 Community
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\" (
    for /f %%i in ('dir /b /ad "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\"') do (
        set "VS_COMPILER=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\%%i\bin\Hostx64\x64"
        set "VS_VERSION=VS2022 Community (%%i)"
        goto :vs_found
    )
)

REM Check for VS2022 Professional
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\" (
    for /f %%i in ('dir /b /ad "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\"') do (
        set "VS_COMPILER=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\%%i\bin\Hostx64\x64"
        set "VS_VERSION=VS2022 Professional (%%i)"
        goto :vs_found
    )
)

REM Check for VS2022 Enterprise
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\" (
    for /f %%i in ('dir /b /ad "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\"') do (
        set "VS_COMPILER=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\%%i\bin\Hostx64\x64"
        set "VS_VERSION=VS2022 Enterprise (%%i)"
        goto :vs_found
    )
)

REM Check for VS2019 BuildTools
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\" (
    for /f %%i in ('dir /b /ad "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\"') do (
        set "VS_COMPILER=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\%%i\bin\Hostx64\x64"
        set "VS_VERSION=VS2019 BuildTools (%%i)"
        goto :vs_found
    )
)

REM Check for VS2019 Community
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\" (
    for /f %%i in ('dir /b /ad "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\"') do (
        set "VS_COMPILER=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\%%i\bin\Hostx64\x64"
        set "VS_VERSION=VS2019 Community (%%i)"
        goto :vs_found
    )
)

REM Check for VS2019 Professional
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\" (
    for /f %%i in ('dir /b /ad "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\"') do (
        set "VS_COMPILER=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\%%i\bin\Hostx64\x64"
        set "VS_VERSION=VS2019 Professional (%%i)"
        goto :vs_found
    )
)

REM Check for VS2019 Enterprise
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\" (
    for /f %%i in ('dir /b /ad "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\"') do (
        set "VS_COMPILER=C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\%%i\bin\Hostx64\x64"
        set "VS_VERSION=VS2019 Enterprise (%%i)"
        goto :vs_found
    )
)

REM Check for VS2017
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\" (
    for /f %%i in ('dir /b /ad "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\"') do (
        set "VS_COMPILER=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\%%i\bin\Hostx64\x64"
        set "VS_VERSION=VS2017 Community (%%i)"
        goto :vs_found
    )
)

echo ERROR: Visual Studio not found!
echo Please install Visual Studio from: https://visualstudio.microsoft.com/downloads/
echo.
pause
exit /b 1

:vs_found
echo [OK] Found %VS_VERSION%
echo     Compiler path: %VS_COMPILER%
echo.

REM ============================================================================
REM STEP 3: Determine compatibility and set flags
REM ============================================================================
echo [3/3] Checking compatibility...

REM CUDA 12.x with VS2022 needs -allow-unsupported-compiler flag
echo !CUDA_VERSION! | findstr /C:"12." >nul
if !errorlevel! equ 0 (
    echo !VS_VERSION! | findstr /C:"VS2022" >nul
    if !errorlevel! equ 0 (
        set "EXTRA_FLAGS=-allow-unsupported-compiler"
        echo [INFO] CUDA 12.x with VS2022 detected - adding -allow-unsupported-compiler flag
    )
)

REM CUDA 11.x with VS2022 also needs -allow-unsupported-compiler flag
echo !CUDA_VERSION! | findstr /C:"11." >nul
if !errorlevel! equ 0 (
    echo !VS_VERSION! | findstr /C:"VS2022" >nul
    if !errorlevel! equ 0 (
        set "EXTRA_FLAGS=-allow-unsupported-compiler"
        echo [INFO] CUDA 11.x with VS2022 detected - adding -allow-unsupported-compiler flag
    )
)

echo.
echo ============================================================================
echo Configuration Summary:
echo   CUDA Version: %CUDA_VERSION%
echo   Visual Studio: %VS_VERSION%
echo   Extra Flags: %EXTRA_FLAGS%
echo ============================================================================
echo.

REM ============================================================================
REM STEP 4: Build the project
REM ============================================================================
echo Starting build...
echo.
echo Executing: "%NVCC_PATH%" main.cu -o gpuff.exe -ccbin "%VS_COMPILER%" %EXTRA_FLAGS%
echo.

"%NVCC_PATH%" main.cu -o gpuff.exe -ccbin "%VS_COMPILER%" %EXTRA_FLAGS%

if !errorlevel! neq 0 (
    echo.
    echo ============================================================================
    echo BUILD FAILED!
    echo ============================================================================
    echo.
    echo Troubleshooting tips:
    echo   1. Make sure all CUDA files are in the current directory
    echo   2. Check that your GPU supports the CUDA compute capability
    echo   3. Try cleaning previous build artifacts: del *.exe *.exp *.lib
    echo   4. Check for compiler errors above
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo BUILD SUCCESSFUL!
echo ============================================================================
echo.
echo Generated files:
if exist gpuff.exe (
    echo   [OK] gpuff.exe - Main executable
    for %%A in (gpuff.exe) do echo        Size: %%~zA bytes
)
if exist gpuff.lib echo   [OK] gpuff.lib - Import library
if exist gpuff.exp echo   [OK] gpuff.exp - Export file
echo.
echo Environment used:
echo   CUDA: %CUDA_VERSION%
echo   Compiler: %VS_VERSION%
echo.
echo To run the program: gpuff.exe
echo ============================================================================
echo.

REM Ask if user wants to run the program
set /p RUN_NOW="Do you want to run gpuff.exe now? (y/n): "
if /i "%RUN_NOW%"=="y" (
    echo.
    echo Running gpuff.exe...
    echo ----------------------------------------------------------------------------
    gpuff.exe
    echo ----------------------------------------------------------------------------
    echo.
)

pause
endlocal