@echo off
rem Pilot on the development tree: fine mode 4 only at ROI widths from a threshold up (not formal).
cd /d C:\r-cuda\gpu4
if not exist pilot4 mkdir pilot4
if exist pilot4\done.txt del pilot4\done.txt
set B=C:\r-cuda\gpu4\build\experiments\fpm_frame_bench.exe
set T=C:\r-cuda\gpu4\wt\cpp-core\tools\benchmark_frame_backend.py
for %%W in (0 64 128 256 512) do (
  set FPM_PILOT_FINE_WIDTH=%%W
  start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe %T% --bench %B% --pgm-dir C:\r51\bundle\pgm --cases Src1,Src2,Src3,Src4,Src6,Src7,Src8,Src9,Src10,Src5-0 --modes subpixel --arm best=2,1,256,3,3,2 --arm f4=2,1,256,3,3,4 --rounds 3 --repeat 5 --warmup-ms 300 --output pilot4\w%%W.json > pilot4\w%%W.log 2>&1
)
echo PILOTDONE > pilot4\done.txt
