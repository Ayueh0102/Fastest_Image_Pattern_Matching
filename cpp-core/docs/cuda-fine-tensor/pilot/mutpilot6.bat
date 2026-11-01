@echo off
rem Mutant pilot on the development tree, then one Nsight Compute run with three blocks per SM (not formal).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_ROOT%\bin;%PATH%"
cd /d C:\r-cuda\gpu5
if exist pilot6\mutdone.txt del pilot6\mutdone.txt
C:\r50\venv\Scripts\python.exe C:\r-cuda\gpu5\wt\cpp-core\tools\cuda_mutants.py C:\r-cuda\gpu5\build C:\r-cuda\gpu5\wt pilot6\mutpilot.json M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,M16,U10,U11,U12,K8,U9 > pilot6\mutpilot.log 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='wt\cpp-core\experiments\coarse_cuda.cu'; $o=[IO.File]::ReadAllText($p); [IO.File]::WriteAllText('occ-original.cu',$o); [IO.File]::WriteAllText($p,$o.Replace('perSm / 2 - 1024','perSm / 3 - 1024'))"
cmake --build build > pilot6\occ-build.log 2>&1
set "NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat"
call "%NCU%" --kernel-name regex:fineStripes --launch-skip 6 --launch-count 1 --set full -f -o pilot6\occ3 build\experiments\fpm_frame_bench.exe 2 1 256 6 subpixel 3 0 C:\r51\bundle\pgm\Src6.pgm C:\r51\bundle\pgm\Dst6.pgm 15 0.8 180 0 256 3 5 > pilot6\occ3.stdout.txt 2>&1
call "%NCU%" --import pilot6\occ3.ncu-rep --page details > pilot6\occ3.txt 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -Command "[IO.File]::WriteAllText('wt\cpp-core\experiments\coarse_cuda.cu',[IO.File]::ReadAllText('occ-original.cu'))"
cmake --build build >> pilot6\occ-build.log 2>&1
echo MUTDONE > pilot6\mutdone.txt
