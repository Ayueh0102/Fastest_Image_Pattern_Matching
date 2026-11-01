@echo off
rem Pilot on the development tree: fine mode 5 from a ROI width up, then shared memory per block (not formal).
cd /d C:\r-cuda\gpu5
if not exist pilot5 mkdir pilot5
if exist pilot5\done.txt del pilot5\done.txt
set B=C:\r-cuda\gpu5\build\experiments\fpm_frame_bench.exe
set T=C:\r-cuda\gpu5\wt\cpp-core\tools\benchmark_frame_backend.py
set C=Src1,Src2,Src3,Src4,Src6,Src7,Src8,Src9,Src10,Src5-0
for %%W in (0 64 128 256 512) do (
  call set FPM_PILOT_FINE_TENSOR_WIDTH=%%W
  start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe %T% --bench %B% --pgm-dir C:\r51\bundle\pgm --cases %C% --modes exhaustive,subpixel --arm k6=2,1,256,6,3,4 --arm m=2,1,256,6,3,5 --rounds 3 --repeat 5 --warmup-ms 300 --output pilot5\w%%W.json > pilot5\w%%W.log 2>&1
)
set FPM_PILOT_FINE_TENSOR_WIDTH=
for %%D in (3 4) do (
  set FPM_PILOT_FINE_SM_DIV=%%D
  for %%M in (16 8) do (
    set FPM_PILOT_FINE_MIN_STRIPE=%%M
    start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe %T% --bench %B% --pgm-dir C:\r51\bundle\pgm --cases %C% --modes exhaustive,subpixel --arm k6=2,1,256,6,3,4 --arm m=2,1,256,6,3,5 --rounds 3 --repeat 5 --warmup-ms 300 --output pilot5\d%%D-m%%M.json > pilot5\d%%D-m%%M.log 2>&1
  )
)
echo PILOTDONE > pilot5\done.txt
