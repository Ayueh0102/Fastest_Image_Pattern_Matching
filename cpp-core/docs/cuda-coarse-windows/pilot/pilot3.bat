@echo off
rem Pilot on the development tree: coarse modes with fine mode 4, warps per block 4 and 8, then larger templates (not formal).
cd /d C:\r-cuda\gpu4
if not exist pilot3 mkdir pilot3
if exist pilot3\done.txt del pilot3\done.txt
set B=C:\r-cuda\gpu4\build\experiments\fpm_frame_bench.exe
set T=C:\r-cuda\gpu4\wt\cpp-core\tools\benchmark_frame_backend.py
for %%W in (4 8 2) do (
  set FPM_PILOT_COARSE_WARPS=%%W
  start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe %T% --bench %B% --pgm-dir C:\r51\bundle\pgm --cases Src1,Src2,Src3,Src4,Src6,Src7,Src8,Src9,Src10,Src5-0 --arm best=2,1,256,3,3,2 --arm c3f4=2,1,256,3,3,4 --arm c5f4=2,1,256,5,3,4 --arm c6f4=2,1,256,6,3,4 --rounds 2 --repeat 5 --warmup-ms 300 --output pilot3\w%%W.json > pilot3\w%%W.log 2>&1
)
set FPM_PILOT_COARSE_WARPS=
start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe C:\r-cuda\gpu4\pilotbig.py > pilot3\big.log 2>&1
echo PILOTDONE > pilot3\done.txt
