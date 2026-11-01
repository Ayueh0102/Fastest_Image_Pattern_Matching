@echo off
rem Pilot of the mutants the threshold test should kill, and of the counter anchors (not a gate).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_ROOT%\bin;%PATH%"
cd /d C:\r-cuda\gpu4
C:\r50\venv\Scripts\python.exe C:\r-cuda\gpu4\wt\cpp-core\tools\cuda_mutants.py C:\r-cuda\gpu4\build C:\r-cuda\gpu4\wt C:\r-cuda\gpu4\mutpilot2.json L22,L13,K10,K12 > C:\r-cuda\gpu4\mutpilot2.log 2>&1
echo MUTDONE %errorlevel% >> C:\r-cuda\gpu4\mutpilot2.log
