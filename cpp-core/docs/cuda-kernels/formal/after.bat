@echo off
rem After the formal timing: Nsight profiles, then compiled mutants on a separate copy of c67ef78.
set F=C:\r-cuda\gpu3\formal
cd /d %F%
echo after-start %date% %time% > logs\after.log
powershell -NoProfile -ExecutionPolicy Bypass -File %F%\profile.ps1 -Bench %F%\build-after-cuda\experiments\fpm_frame_bench.exe -Out %F%\profile > logs\profile.log 2>&1
echo profile %errorlevel% >> logs\after.log
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_ROOT%\bin;%PATH%"
if not exist mut mkdir mut
tar -xf %F%\src-after.tar -C mut
cmake -S %F%\mut\src-after\cpp-core -B %F%\mut-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_CUDA=ON "-DCMAKE_CUDA_COMPILER=%CUDA_ROOT%\bin\nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_ROOT%" -DCMAKE_CUDA_ARCHITECTURES=120 > logs\mut-configure.log 2>&1
cmake --build %F%\mut-build > logs\mut-build.log 2>&1
echo mut-build %errorlevel% >> logs\after.log
C:\r50\venv\Scripts\python.exe %F%\mut\src-after\cpp-core\tools\cuda_mutants.py %F%\mut-build %F%\mut\src-after results\mutants-c67ef78.json > logs\mutants.log 2>&1
echo mutants %errorlevel% >> logs\after.log
echo AFTERDONE %date% %time% >> logs\after.log
