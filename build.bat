@echo off
REM ============================================================================
REM GPUFF-RCAPv3 Universal Build Script
REM ============================================================================
REM Works from ANY command prompt (cmd, PowerShell, Git Bash, VS prompt, etc.)
REM Automatically detects and initializes Visual Studio environment
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo GPUFF-RCAPv3 Build System
echo ============================================================================
echo.

REM ----------------------------------------------------------------------------
REM Auto-detect Visual Studio Installation
REM ----------------------------------------------------------------------------
echo [1/4] Auto-detecting Visual Studio installation...

set "VS_FOUND=0"
set "VS_PATH="

REM Try vswhere.exe (official VS locator tool)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VS_INSTALL_PATH=%%i"
    )
    if defined VS_INSTALL_PATH (
        set "VS_PATH=!VS_INSTALL_PATH!\VC\Auxiliary\Build\vcvarsall.bat"
        if exist "!VS_PATH!" (
            echo Found Visual Studio via vswhere: !VS_INSTALL_PATH!
            set "VS_FOUND=1"
        )
    )
)

REM Fallback: Try common VS 2022 locations
if "%VS_FOUND%"=="0" (
    echo vswhere not found, trying common locations...

    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    if exist "!VS_PATH!" (
        echo Found Visual Studio 2022 Community
        set "VS_FOUND=1"
    )
)

if "%VS_FOUND%"=="0" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    if exist "!VS_PATH!" (
        echo Found Visual Studio 2022 Professional
        set "VS_FOUND=1"
    )
)

if "%VS_FOUND%"=="0" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    if exist "!VS_PATH!" (
        echo Found Visual Studio 2022 Enterprise
        set "VS_FOUND=1"
    )
)

REM Fallback: Try VS 2019
if "%VS_FOUND%"=="0" (
    set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"
    if exist "!VS_PATH!" (
        echo Found Visual Studio 2019 Community
        set "VS_FOUND=1"
    )
)

if "%VS_FOUND%"=="0" (
    echo.
    echo ERROR: Visual Studio not found!
    echo.
    echo Please install Visual Studio 2019 or 2022 with C++ support:
    echo https://visualstudio.microsoft.com/downloads/
    echo.
    echo Required workload: "Desktop development with C++"
    echo.
    pause
    exit /b 1
)

echo.

REM ----------------------------------------------------------------------------
REM Check if already in VS environment (avoid double initialization)
REM ----------------------------------------------------------------------------
echo [2/4] Checking compiler environment...

where cl.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo Visual Studio environment already initialized (cl.exe found in PATH^)
    echo Skipping vcvarsall.bat call...
    goto :compile
)

REM ----------------------------------------------------------------------------
REM Initialize Visual Studio Environment
REM ----------------------------------------------------------------------------
echo Initializing Visual Studio environment...
echo Running: "%VS_PATH%" x64
echo.

call "%VS_PATH%" x64 >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to initialize Visual Studio environment
    echo.
    pause
    exit /b 1
)

echo Visual Studio environment initialized successfully.
echo.

:compile

REM ----------------------------------------------------------------------------
REM Verify CUDA is available
REM ----------------------------------------------------------------------------
echo [3/4] Verifying CUDA Toolkit...

where nvcc.exe >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: CUDA Toolkit not found!
    echo.
    echo Please install NVIDIA CUDA Toolkit 12.2 or later:
    echo https://developer.nvidia.com/cuda-downloads
    echo.
    echo After installation, ensure CUDA bin directory is in PATH:
    echo Example: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('nvcc --version ^| findstr "release"') do (
    echo Found CUDA: %%i
)
echo.

REM ----------------------------------------------------------------------------
REM Compile
REM ----------------------------------------------------------------------------
echo [4/4] Compiling CUDA source files...
echo.
echo Command: nvcc main.cu -o gpuff.exe -allow-unsupported-compiler -rdc=true
echo.

nvcc main.cu -o gpuff.exe -allow-unsupported-compiler -rdc=true

if errorlevel 1 (
    echo.
    echo ============================================================================
    echo BUILD FAILED
    echo ============================================================================
    echo.
    echo Check the error messages above for details.
    echo.
    echo Common issues:
    echo   1. Missing source files (main.cu, *.cuh)
    echo   2. Syntax errors in CUDA code
    echo   3. Incompatible CUDA/Visual Studio versions
    echo.
    echo For detailed troubleshooting, see: docs\TROUBLESHOOTING.md
    echo.
    pause
    exit /b 1
)

REM ----------------------------------------------------------------------------
REM Success
REM ----------------------------------------------------------------------------
echo.
echo ============================================================================
echo BUILD SUCCESSFUL
echo ============================================================================
echo.
echo Output files:
if exist gpuff.exe (
    for %%A in (gpuff.exe) do echo   - gpuff.exe  ^(%%~zA bytes^)
)
if exist gpuff.lib (
    for %%A in (gpuff.lib) do echo   - gpuff.lib  ^(%%~zA bytes^)
)
if exist gpuff.exp (
    for %%A in (gpuff.exp) do echo   - gpuff.exp  ^(%%~zA bytes^)
)
echo.
echo ----------------------------------------------------------------------------
echo To run the simulation:
echo ----------------------------------------------------------------------------
echo   1. Ensure input files are in .\input\RCAPdata\
echo      Required: Test1.inp, MACCS60.NDL, MACCS_DCF_New2.LIB, METEO.inp
echo.
echo   2. Execute: gpuff.exe
echo.
echo   3. Output will be generated in:
echo      - .\output\  ^(puff visualization^)
echo      - .\evac\    ^(evacuee tracking^)
echo.
echo For more information, see: docs\QUICK_START.md
echo ============================================================================
echo.

endlocal
