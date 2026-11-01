@echo off
rem Pilot on the development tree: fine mode 4 at several column tile widths (not formal).
cd /d C:\r-cuda\gpu4
if not exist pilott mkdir pilott
if exist pilott\done.txt del pilott\done.txt
set B=C:\r-cuda\gpu4\build\experiments\fpm_frame_bench.exe
set T=C:\r-cuda\gpu4\wt\cpp-core\tools\benchmark_frame_backend.py
for %%W in (0 256 128 64) do (
  set FPM_PILOT_FINE_TILE=%%W
  start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe %T% --bench %B% --pgm-dir C:\r51\bundle\pgm --cases Src1,Src2,Src3,Src4,Src6,Src7,Src8,Src9,Src10,Src5-0 --modes exhaustive,subpixel --arm best=2,1,256,3,3,2 --arm c6f2=2,1,256,6,3,2 --arm c6f4=2,1,256,6,3,4 --arm c5f4=2,1,256,5,3,4 --rounds 2 --repeat 5 --warmup-ms 300 --output pilott\t%%W.json > pilott\t%%W.log 2>&1
)
echo PILOTDONE > pilott\done.txt
