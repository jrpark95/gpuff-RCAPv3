@echo off
echo ============================================================================
echo Building GPUFF-RCAPv3 Working Version
echo ============================================================================

set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe"
set "VS2022_COMPILER=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64"

echo Compiling gpuff_working.cu...
"%NVCC%" gpuff_working.cu -o gpuff.exe -ccbin "%VS2022_COMPILER%" -allow-unsupported-compiler

if errorlevel 1 (
    echo BUILD FAILED
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo BUILD SUCCESSFUL!
echo ============================================================================
echo.
echo To run: gpuff.exe
echo.
pause