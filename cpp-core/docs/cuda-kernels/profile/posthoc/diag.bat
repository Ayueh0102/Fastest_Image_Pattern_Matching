@echo off
rem Waits for after.bat (mutants) to finish, then runs the post-hoc Nsight diagnosis.
set F=C:\r-cuda\gpu3\formal
:wait
findstr /c:"AFTERDONE" %F%\logs\after.log >nul 2>&1
if errorlevel 1 (timeout /t 30 /nobreak >nul & goto wait)
powershell -NoProfile -ExecutionPolicy Bypass -File C:\r-cuda\gpu3\diag\diag.ps1 -Bench %F%\build-after-cuda\experiments\fpm_frame_bench.exe -Out C:\r-cuda\gpu3\diag\out > C:\r-cuda\gpu3\diag\diag.log 2>&1
echo DIAGDONE %date% %time% >> C:\r-cuda\gpu3\diag\diag.log
