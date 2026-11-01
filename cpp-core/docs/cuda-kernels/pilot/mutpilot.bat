@echo off
rem Pilot of the kernel-mode mutants on the development tree (not a gate).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_ROOT%\bin;%PATH%"
cd /d C:\r-cuda\gpu3
C:\r50\venv\Scripts\python.exe C:\r-cuda\wt-kern1\cpp-core\tools\cuda_mutants.py C:\r-cuda\build-wt-kern1 C:\r-cuda\wt-kern1 C:\r-cuda\gpu3\mutpilot.json K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12,K13,U5,U6,T6,C1,C2,W4 > C:\r-cuda\gpu3\mutpilot.log 2>&1
echo MUTDONE %errorlevel% >> C:\r-cuda\gpu3\mutpilot.log
