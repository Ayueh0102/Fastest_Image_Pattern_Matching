@echo off
cd /d C:\r-cuda\gpu3
start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe C:\r-cuda\wt-kern1\cpp-core\tools\benchmark_frame_backend.py --bench C:\r-cuda\build-wt-kern1\experiments\fpm_frame_bench.exe --pgm-dir C:\r51\bundle\pgm --arm cpu=0,0,0,0 --arm m3=2,1,256,0,3 --arm c2=2,1,256,2,3 --arm c3=2,1,256,3,3 --arm f2=2,1,256,0,3,2 --arm u7=2,1,256,0,7 --arm t4=2,1,256,0,3,0,4 --arm all3=2,1,256,3,3,2,4 --arm all7=2,1,256,3,7,2,4 --arm fp32=2,1,256,4,3,3 --inexact fp32 --rounds 2 --repeat 5 --warmup-ms 500 --output C:\r-cuda\gpu3\pilot1.json > C:\r-cuda\gpu3\pilot1.log 2>&1
echo PILOTDONE >> C:\r-cuda\gpu3\pilot1.log
