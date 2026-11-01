@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_ROOT%\bin;%PATH%"
cd /d C:\r-cuda\gpu4\probe
nvcc -arch=sm_120 -O2 mma_probe.cu -o mma_probe.exe > build.log 2>&1
mma_probe.exe > probe.txt 2>&1
echo PROBE %errorlevel% >> build.log
