# Post-hoc diagnosis after the formal runs (not registered, not formal): Nsight Systems of
# m0, m3 and u7 on Src6 subpixel (fine kernel time with requests read from mapped memory
# versus uploaded) and of m3 and t4 on Src5-0 and Src1 exhaustive (where the copy threads' time goes).
param([string]$Bench, [string]$Out)
$ErrorActionPreference = "Continue"
$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe"
$p = "C:\r51\bundle\pgm"
New-Item -ItemType Directory -Force $Out | Out-Null
# name, source, template, num, score, angle, overlap, minarea, mode, coarse, transfer, fine, threads
$runs = @(
  @("Src6-subpixel-m0","Src6","Dst6","15","0.8","180","0","256","subpixel","0","0","0","1"),
  @("Src6-subpixel-m3","Src6","Dst6","15","0.8","180","0","256","subpixel","0","3","0","1"),
  @("Src6-subpixel-u7","Src6","Dst6","15","0.8","180","0","256","subpixel","0","7","0","1"),
  @("Src5-0-exhaustive-m3","Src5-0","Dst5","1","0.8","180","0","256","exhaustive","0","3","0","1"),
  @("Src5-0-exhaustive-t4","Src5-0","Dst5","1","0.8","180","0","256","exhaustive","0","3","0","4"),
  @("Src1-exhaustive-m3","Src1","Dst1","5","0.8","180","0","256","exhaustive","0","3","0","1"),
  @("Src1-exhaustive-t4","Src1","Dst1","5","0.8","180","0","256","exhaustive","0","3","0","4"))
foreach ($r in $runs) {
  $n = $r[0]
  cmd /c "`"$nsys`" profile --trace=cuda,nvtx,wddm --force-overwrite=true -o $Out\$n $Bench 2 1 256 $($r[9]) $($r[8]) 20 1000 $p\$($r[1]).pgm $p\$($r[2]).pgm $($r[3]) $($r[4]) $($r[5]) $($r[6]) $($r[7]) $($r[10]) $($r[11]) $($r[12]) > $Out\$n.stdout.txt 2>&1"
  foreach ($rep in "cuda_gpu_kern_sum","nvtx_sum","cuda_api_sum","cuda_gpu_mem_time_sum","cuda_gpu_trace") {
    cmd /c "`"$nsys`" stats --report $rep --format csv --force-export=true $Out\$n.nsys-rep > $Out\$n.$rep.csv 2>nul"
  }
}
"DIAGDONE" | Out-File $Out\done.txt
