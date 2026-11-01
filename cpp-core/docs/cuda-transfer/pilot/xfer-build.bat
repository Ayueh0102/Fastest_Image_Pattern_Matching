@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_ROOT%\bin;%PATH%"
cd /d C:\r-cuda\gpu2\pilot
nvcc -O2 -arch=sm_120 -Xcompiler=/utf-8 xfer_pilot.cu -o xfer_pilot.exe
