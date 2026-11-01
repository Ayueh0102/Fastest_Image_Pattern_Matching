@echo off
rem Formal kernel-mode gates and timing (protocol ea7b229). Each step runs in its own cmd.
set F=C:\r-cuda\gpu3\formal
set PY=C:\r50\venv\Scripts\python.exe
set PGM=C:\r51\bundle\pgm
set B=%F%\build-after-cuda\experiments\fpm_frame_bench.exe
set T=%F%\src-after\cpp-core\tools\benchmark_frame_backend.py
cd /d %F%
if not exist logs mkdir logs
if not exist results mkdir results
echo start %date% %time% > logs\formal.log
cmd /c %F%\build-cuda.bat
echo step1 %errorlevel% >> logs\formal.log
%PY% %F%\check_pgm.py %F%\pgm-manifest.json %PGM% > logs\pgm-check.log 2>&1
echo step2 %errorlevel% >> logs\formal.log
%F%\build-after-cuda\experiments\fpm_cuda_backend_test.exe > logs\cuda-backend-test.log 2>&1
echo step3a %errorlevel% >> logs\formal.log
%F%\build-after-cuda\experiments\fpm_cuda_kernel_test.exe > logs\cuda-kernel-test.log 2>&1
echo step3b %errorlevel% >> logs\formal.log
%PY% %T% --bench %B% --pgm-dir %PGM% --arm cpu=0,0,0,0 --arm c3_batch8=2,1,8,3,3,2 --arm c3_mode0=2,1,256,3,0,2 --arm c3_cpu_fine=2,1,0,3,1 --arm f2_threads=2,1,256,0,3,2,4 --rounds 2 --repeat 1 --warmup-ms 0 --output results\identity-ablations.json > logs\identity-ablations.log 2>&1
echo step4 %errorlevel% >> logs\formal.log
echo timing-start %date% %time% >> logs\formal.log
start /wait /b /affinity C03C02 %PY% %T% --bench %B% --pgm-dir %PGM% --arm cpu=0,0,0,0 --arm m3=2,1,256,0,3 --arm c2=2,1,256,2,3 --arm c3=2,1,256,3,3 --arm f2=2,1,256,0,3,2 --arm best=2,1,256,3,3,2 --arm u7=2,1,256,0,7 --arm t4=2,1,256,0,3,0,4 --arm fp32=2,1,256,4,3,3 --inexact fp32 --rounds 8 --repeat 5 --warmup-ms 500 --output results\timing.json > logs\timing.log 2>&1
echo step5 %errorlevel% >> logs\formal.log
echo FORMALDONE %date% %time% >> logs\formal.log
