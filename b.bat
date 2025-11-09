@echo off
echo ============================================================================
echo GPUFF-RCAPv3 Build Script (Current Environment)
echo Using: CUDA 12.2 + VS2022 Community
echo ============================================================================
echo.

REM Current environment configuration
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe"
set "VS2022_COMPILER=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64"

echo Checking dependencies...
if not exist "%NVCC%" (
    echo ERROR: CUDA 12.2 not found
    echo Install from: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

if not exist "%VS2022_COMPILER%\cl.exe" (
    echo ERROR: VS2022 Community not found
    echo Install from: https://visualstudio.microsoft.com/downloads/
    pause
    exit /b 1
)

echo [OK] CUDA 12.2 found
echo [OK] VS2022 Community found
echo.
echo Compiling...
echo.

"%NVCC%" main.cu -o gpuff.exe -ccbin "%VS2022_COMPILER%" -allow-unsupported-compiler

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
