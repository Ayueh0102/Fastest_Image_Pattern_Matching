@echo off
rem Formal fine Tensor core gates and timing (registered protocol). Each step runs in its own cmd.
set F=C:\r-cuda\gpu5\formal
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
%PY% %T% --bench %B% --pgm-dir %PGM% --arm cpu=0,0,0,0 --arm t5_batch8=2,1,8,6,3,5 --arm t5_mode0=2,1,256,6,0,5 --arm t5_c5=2,1,256,5,3,5 --arm t5_c3=2,1,256,3,0,5 --arm t5_threads=2,1,256,6,3,5,4 --rounds 2 --repeat 1 --warmup-ms 0 --output results\identity-standard.json > logs\identity-standard.log 2>&1
echo step4 %errorlevel% >> logs\formal.log
%PY% %T% --bench %B% --pgm-dir %PGM% --min-area-scale 16 --modes exhaustive,subpixel --arm cpu=0,0,0,0 --arm b2=2,1,256,3,3,2 --arm t5=2,1,256,6,3,5 --rounds 2 --repeat 1 --warmup-ms 0 --output results\identity-shallow.json > logs\identity-shallow.log 2>&1
echo step5 %errorlevel% >> logs\formal.log
echo timing-start %date% %time% >> logs\formal.log
start /wait /b /affinity C03C02 %PY% %T% --bench %B% --pgm-dir %PGM% --arm cpu=0,0,0,0 --arm b2=2,1,256,3,3,2 --arm k6=2,1,256,6,3,4 --arm t5=2,1,256,6,3,5 --arm f32=2,1,256,7,3,6 --arm old32=2,1,256,4,3,3 --inexact f32 --inexact old32 --rounds 8 --repeat 5 --warmup-ms 500 --output results\timing-standard.json > logs\timing-standard.log 2>&1
echo step6 %errorlevel% >> logs\formal.log
start /wait /b /affinity C03C02 %PY% %T% --bench %B% --pgm-dir %PGM% --min-area-scale 16 --modes exhaustive,subpixel --arm b2=2,1,256,3,3,2 --arm k6=2,1,256,6,3,4 --arm t5=2,1,256,6,3,5 --arm f32=2,1,256,7,3,6 --inexact f32 --rounds 8 --repeat 5 --warmup-ms 500 --output results\timing-shallow.json > logs\timing-shallow.log 2>&1
echo step7 %errorlevel% >> logs\formal.log
echo FORMALDONE %date% %time% >> logs\formal.log
