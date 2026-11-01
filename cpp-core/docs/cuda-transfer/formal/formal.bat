@echo off
rem Formal transfer-mode gates and timing (protocol 0cbf3f6). Each step runs in its own cmd.
set F=C:\r-cuda\gpu2\formal
set PY=C:\r50\venv\Scripts\python.exe
set PGM=C:\r51\bundle\pgm
set B=%F%\build-after-cuda\experiments\fpm_frame_bench.exe
set T=%F%\src-after\cpp-core\tools\benchmark_frame_backend.py
cd /d %F%
if not exist logs mkdir logs
if not exist results mkdir results
echo start %date% %time% > logs\formal.log
rem 1. CUDA build of c1068b2 and ctest
cmd /c %F%\build-cuda.bat
echo step1 %errorlevel% >> logs\formal.log
rem 2. inputs
%PY% %F%\check_pgm.py %F%\pgm-manifest.json %PGM% > logs\pgm-check.log 2>&1
echo step2 %errorlevel% >> logs\formal.log
rem 3. CUDA test outputs kept as evidence
%F%\build-after-cuda\experiments\fpm_cuda_backend_test.exe > logs\cuda-backend-test.log 2>&1
echo step3a %errorlevel% >> logs\formal.log
%F%\build-after-cuda\experiments\fpm_cuda_kernel_test.exe > logs\cuda-kernel-test.log 2>&1
echo step3b %errorlevel% >> logs\formal.log
rem 4. identity-only arms
%PY% %T% --bench %B% --pgm-dir %PGM% --arm cpu=0,0,0,0 --arm m3_batch8=2,1,8,0,3 --arm m3_unfused=2,1,256,1,3 --arm m3_cpu_pyramid=2,0,256,0,3 --arm m1_cpu_fine=2,1,0,0,1 --arm m3_cpu_fine=2,1,0,0,3 --rounds 2 --repeat 1 --warmup-ms 0 --output results\identity-ablations.json > logs\identity-ablations.log 2>&1
echo step4 %errorlevel% >> logs\formal.log
rem 5. formal timing on P-cores
echo timing-start %date% %time% >> logs\formal.log
start /wait /b /affinity C03C02 %PY% %T% --bench %B% --pgm-dir %PGM% --arm cpu=0,0,0,0 --arm m0=2,1,256,0,0 --arm m1=2,1,256,0,1 --arm m2=2,1,256,0,2 --arm m3=2,1,256,0,3 --rounds 8 --repeat 5 --warmup-ms 500 --output results\timing.json > logs\timing.log 2>&1
echo step5 %errorlevel% >> logs\formal.log
echo FORMALDONE %date% %time% >> logs\formal.log
