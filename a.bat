@echo off
echo ============================================================================
echo GPUFF-RCAPv3 Build Script (Working Configuration)
echo Using: CUDA 11.8 + VS2019 Build Tools
echo ============================================================================
echo.

REM Working configuration that successfully builds
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe"
set "VS2019_COMPILER=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"

echo Checking dependencies...
if not exist "%NVCC%" (
    echo ERROR: CUDA 11.8 not found
    echo Install from: https://developer.nvidia.com/cuda-11-8-0-download-archive
    pause
    exit /b 1
)

if not exist "%VS2019_COMPILER%\cl.exe" (
    echo ERROR: VS2019 Build Tools not found
    echo Install from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019
    pause
    exit /b 1
)

echo [OK] CUDA 11.8 found
echo [OK] VS2019 Build Tools found
echo.
echo Compiling...
echo.

"%NVCC%" main.cu -o gpuff.exe -ccbin "%VS2019_COMPILER%"

if errorlevel 1 (
    echo.
    echo BUILD FAILED
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo BUILD SUCCESSFUL!
echo ============================================================================
echo.
echo Generated files:
if exist gpuff.exe echo   - gpuff.exe (executable)
if exist gpuff.lib echo   - gpuff.lib (library)
if exist gpuff.exp echo   - gpuff.exp (exports)
echo.
echo To run: gpuff.exe
echo ============================================================================
echo.
pause