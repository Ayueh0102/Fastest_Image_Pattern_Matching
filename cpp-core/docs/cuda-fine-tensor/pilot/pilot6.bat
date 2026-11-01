@echo off
rem Pilot on the development tree: all-binary32 ablation (coarse 7, fine 6) against the exact arm, and the row statistics on the fixtures (not formal).
cd /d C:\r-cuda\gpu5
if not exist pilot6 mkdir pilot6
if exist pilot6\done.txt del pilot6\done.txt
set B=C:\r-cuda\gpu5\build\experiments\fpm_frame_bench.exe
set T=C:\r-cuda\gpu5\wt\cpp-core\tools\benchmark_frame_backend.py
set C=Src1,Src2,Src3,Src4,Src6,Src7,Src8,Src9,Src10,Src5-0
start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe %T% --bench %B% --pgm-dir C:\r51\bundle\pgm --cases %C% --arm cpu=0,0,0,0 --arm k6=2,1,256,6,3,4 --arm x=2,1,256,6,3,5 --arm y32=2,1,256,7,3,6 --arm old32=2,1,256,4,3,3 --inexact y32 --inexact old32 --rounds 3 --repeat 5 --warmup-ms 300 --output pilot6\standard.json > pilot6\standard.log 2>&1
start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe %T% --bench %B% --pgm-dir C:\r51\bundle\pgm --cases Src1,Src5-0,Src6,Src7,Src10,Src3 --min-area-scale 16 --modes exhaustive,subpixel --arm k6=2,1,256,6,3,4 --arm x=2,1,256,6,3,5 --arm y32=2,1,256,7,3,6 --inexact y32 --rounds 3 --repeat 5 --warmup-ms 300 --output pilot6\shallow.json > pilot6\shallow.log 2>&1
C:\r50\venv\Scripts\python.exe C:\r-cuda\gpu5\wt\cpp-core\tools\cuda_mutants.py C:\r-cuda\gpu5\build C:\r-cuda\gpu5\wt pilot6\mutpilot.json M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,M16,U10,U11,U12,K8,U9 > pilot6\mutpilot.log 2>&1
echo PILOTDONE > pilot6\done.txt
