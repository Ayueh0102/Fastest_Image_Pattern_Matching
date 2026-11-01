@echo off
rem Development build of the working tree copy (not a gate).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_ROOT%\bin;%PATH%"
cd /d C:\r-cuda\gpu5
if not exist build\build.ninja cmake -S wt\cpp-core -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_CUDA=ON "-DCMAKE_CUDA_COMPILER=%CUDA_ROOT%\bin\nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_ROOT%" -DCMAKE_CUDA_ARCHITECTURES=120 > build-configure.log 2>&1
cmake --build build > build.log 2>&1
echo BUILD %errorlevel% >> build.log
