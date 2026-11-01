@echo off
rem Pilot on the development tree: threads per block of fine mode 5 (not formal).
cd /d C:\r-cuda\gpu5
if not exist pilot7 mkdir pilot7
if exist pilot7\done.txt del pilot7\done.txt
set B=C:\r-cuda\gpu5\build\experiments\fpm_frame_bench.exe
set T=C:\r-cuda\gpu5\wt\cpp-core\tools\benchmark_frame_backend.py
set C=Src1,Src2,Src3,Src4,Src6,Src7,Src8,Src9,Src10,Src5-0
for %%N in (256 320 384 448) do (
  call set FPM_PILOT_FINE_THREADS=%%N
  start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe %T% --bench %B% --pgm-dir C:\r51\bundle\pgm --cases %C% --arm k6=2,1,256,6,3,4 --arm m=2,1,256,6,3,5 --rounds 3 --repeat 5 --warmup-ms 300 --output pilot7\n%%N.json > pilot7\n%%N.log 2>&1
)
set FPM_PILOT_FINE_THREADS=
echo PILOTDONE > pilot7\done.txt
