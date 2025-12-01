@echo off
echo ============================================================================
echo Building Simple GPUFF Test
echo ============================================================================

set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe"
set "VS2022_COMPILER=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64"

echo Compiling main_simple.cu...
"%NVCC%" main_simple.cu -o gpuff_simple.exe -ccbin "%VS2022_COMPILER%" -allow-unsupported-compiler

if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

echo BUILD SUCCESSFUL!
echo.
echo Running test...
gpuff_simple.exe