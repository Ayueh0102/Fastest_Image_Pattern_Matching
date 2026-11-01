# Nsight Systems profiles of the formal build, transfer modes 0 and 3; not formal timings.
param([string]$Bench, [string]$Out)
$ErrorActionPreference = "Continue"
$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe"
$p = "C:\r51\bundle\pgm"
New-Item -ItemType Directory -Force $Out | Out-Null
$cases = @(@("Src7","Dst7","5","0.8","180","0","256","exhaustive"), @("Src9","Dst9","100","0.5","0","0.5","1024","exhaustive"),
           @("Src6","Dst6","15","0.8","180","0","256","subpixel"), @("Src5-0","Dst5","1","0.8","180","0","256","exhaustive"))
foreach ($c in $cases) {
  foreach ($t in "0","3") {
    $n = "$($c[0])-$($c[7])-m$t"
    cmd /c "`"$nsys`" profile --trace=cuda,nvtx,wddm --force-overwrite=true -o $Out\$n $Bench 2 1 256 0 $($c[7]) 20 1000 $p\$($c[0]).pgm $p\$($c[1]).pgm $($c[2]) $($c[3]) $($c[4]) $($c[5]) $($c[6]) $t > $Out\$n.stdout.txt 2>&1"
    foreach ($rep in "cuda_gpu_kern_sum","nvtx_sum","cuda_api_sum","cuda_gpu_mem_time_sum") {
      cmd /c "`"$nsys`" stats --report $rep --format csv --force-export=true $Out\$n.nsys-rep > $Out\$n.$rep.csv 2>nul"
    }
  }
}
# Short runs for per-operation timelines (overlap of chunk copies, uploads and kernels).
foreach ($t in "0","3") {
  $n = "trace-Src7-exhaustive-m$t"
  cmd /c "`"$nsys`" profile --trace=cuda,nvtx,wddm --force-overwrite=true -o $Out\$n $Bench 2 1 256 0 exhaustive 3 300 $p\Src7.pgm $p\Dst7.pgm 5 0.8 180 0 256 $t > $Out\$n.stdout.txt 2>&1"
  foreach ($rep in "cuda_gpu_trace","cuda_api_trace","nvtx_pushpop_trace") {
    cmd /c "`"$nsys`" stats --report $rep --format csv --force-export=true $Out\$n.nsys-rep > $Out\$n.$rep.csv 2>nul"
  }
}
"PROFILEDONE" | Out-File $Out\done.txt
