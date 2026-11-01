@echo off
cd /d C:\r-cuda\gpu2\pilot
start /wait /b /affinity C03C02 C:\r50\venv\Scripts\python.exe C:\r-cuda\wt-xfer\cpp-core\tools\benchmark_frame_backend.py --bench C:\r-cuda\build-wt-xfer\experiments\fpm_frame_bench.exe --pgm-dir C:\r51\bundle\pgm --arm cpu=0,0,0,0 --arm m0=2,1,256,0,0 --arm m1=2,1,256,0,1 --arm m2=2,1,256,0,2 --arm m3=2,1,256,0,3 --rounds 2 --repeat 5 --warmup-ms 500 --output C:\r-cuda\gpu2\pilot\pilot1.json > C:\r-cuda\gpu2\pilot\pilot1.log 2>&1
