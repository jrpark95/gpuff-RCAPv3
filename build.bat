@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
nvcc main.cu -o gpuff.exe -allow-unsupported-compiler -rdc=true
