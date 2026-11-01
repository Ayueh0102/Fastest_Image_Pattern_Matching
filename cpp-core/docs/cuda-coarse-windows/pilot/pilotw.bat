@echo off
rem Pilot on the development tree: coarse NCC modes 3, 5, 6 at 1, 2, 4 and 8 warps per block (not formal).
cd /d C:\r-cuda\gpu4
if not exist pilotw mkdir pilotw
set B=C:\r-cuda\gpu4\build\experiments\fpm_frame_bench.exe
set T=C:\r-cuda\gpu4\wt\cpp-core\tools\benchmark_frame_backend.py
for %%W in (8 4 2 1) do (
  set FPM_PILOT_COARSE_WARPS=%%W
  start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe %T% --bench %B% --pgm-dir C:\r51\bundle\pgm --cases Src1,Src2,Src3,Src4,Src6,Src7,Src8,Src9,Src10,Src5-0 --modes exhaustive --arm cpu=0,0,0,0 --arm c3=2,1,256,3,3,2 --arm k5=2,1,256,5,3,2 --arm k6=2,1,256,6,3,2 --rounds 2 --repeat 5 --warmup-ms 300 --output pilotw\w%%W.json > pilotw\w%%W.log 2>&1
)
echo PILOTDONE > pilotw\done.txt
