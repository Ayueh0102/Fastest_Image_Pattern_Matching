@echo off
rem Pilot of the transfer-mode mutants on the development tree (not a gate).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_ROOT%\bin;%PATH%"
cd /d C:\r-cuda\gpu2\pilot
C:\r50\venv\Scripts\python.exe C:\r-cuda\wt-xfer3\cpp-core\tools\cuda_mutants.py C:\r-cuda\build-wt-xfer3 C:\r-cuda\wt-xfer3 C:\r-cuda\gpu2\pilot\mutpilot-t7.json T7 > C:\r-cuda\gpu2\pilot\mutpilot-t7.log 2>&1
echo MUTDONE %errorlevel% >> C:\r-cuda\gpu2\pilot\mutpilot-t7.log
