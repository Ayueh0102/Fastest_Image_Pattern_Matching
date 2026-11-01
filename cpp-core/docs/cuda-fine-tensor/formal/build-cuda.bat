@echo off
rem CUDA build of src-after (a6aa630) and ctest; vcvars runs before PATH is extended (separate lines).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_ROOT%\bin;%PATH%"
set F=C:\r-cuda\gpu5\formal
cd /d %F%
cmake -S %F%\src-after\cpp-core -B %F%\build-after-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DFPM_BUILD_EXPERIMENTS=ON -DFPM_WITH_CUDA=ON "-DCMAKE_CUDA_COMPILER=%CUDA_ROOT%\bin\nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_ROOT%" -DCMAKE_CUDA_ARCHITECTURES=120 > logs\after-cuda-configure.log 2>&1
if errorlevel 1 exit /b 1
cmake --build %F%\build-after-cuda > logs\after-cuda-build.log 2>&1
if errorlevel 1 exit /b 2
ctest --test-dir %F%\build-after-cuda --output-on-failure > logs\after-cuda-ctest.log 2>&1
if errorlevel 1 exit /b 3
