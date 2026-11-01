@echo off
rem Last pilot on the development tree: the wide-ROI block size rule, then the new mutants (not formal).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_ROOT%\bin;%PATH%"
cd /d C:\r-cuda\gpu5
if not exist pilot8 mkdir pilot8
if exist pilot8\done.txt del pilot8\done.txt
set B=C:\r-cuda\gpu5\build\experiments\fpm_frame_bench.exe
set T=C:\r-cuda\gpu5\wt\cpp-core\tools\benchmark_frame_backend.py
start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe %T% --bench %B% --pgm-dir C:\r51\bundle\pgm --cases Src1,Src2,Src3,Src4,Src6,Src7,Src8,Src9,Src10,Src5-0 --arm cpu=0,0,0,0 --arm b2=2,1,256,3,3,2 --arm k6=2,1,256,6,3,4 --arm t5=2,1,256,6,3,5 --arm f32=2,1,256,7,3,6 --arm old32=2,1,256,4,3,3 --inexact f32 --inexact old32 --rounds 3 --repeat 5 --warmup-ms 300 --output pilot8\standard.json > pilot8\standard.log 2>&1
start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe %T% --bench %B% --pgm-dir C:\r51\bundle\pgm --cases Src1,Src5-0,Src6,Src7,Src10,Src3 --min-area-scale 16 --modes exhaustive,subpixel --arm k6=2,1,256,6,3,4 --arm t5=2,1,256,6,3,5 --arm f32=2,1,256,7,3,6 --inexact f32 --rounds 3 --repeat 5 --warmup-ms 300 --output pilot8\shallow.json > pilot8\shallow.log 2>&1
C:\r50\venv\Scripts\python.exe C:\r-cuda\gpu5\wt\cpp-core\tools\cuda_mutants.py C:\r-cuda\gpu5\build C:\r-cuda\gpu5\wt pilot8\mutpilot.json M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,M16,M17,U10,U11,U12 > pilot8\mutpilot.log 2>&1
echo PILOTDONE > pilot8\done.txt
