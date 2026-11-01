# Nsight profiles of the formal build (m3 against best; ncu of the new kernels); not formal timings.
param([string]$Bench, [string]$Out)
$ErrorActionPreference = "Continue"
$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat"
$p = "C:\r51\bundle\pgm"
New-Item -ItemType Directory -Force $Out | Out-Null
$cases = @(@("Src1","Dst1","5","0.8","180","0","256","exhaustive"), @("Src9","Dst9","100","0.5","0","0.5","1024","exhaustive"),
           @("Src6","Dst6","15","0.8","180","0","256","subpixel"))
$arms = @(@("m3","0","3","0"), @("best","3","3","2"))
foreach ($c in $cases) {
  foreach ($a in $arms) {
    $n = "$($c[0])-$($c[7])-$($a[0])"
    cmd /c "`"$nsys`" profile --trace=cuda,nvtx,wddm --force-overwrite=true -o $Out\$n $Bench 2 1 256 $($a[1]) $($c[7]) 20 1000 $p\$($c[0]).pgm $p\$($c[1]).pgm $($c[2]) $($c[3]) $($c[4]) $($c[5]) $($c[6]) $($a[2]) $($a[3]) > $Out\$n.stdout.txt 2>&1"
    foreach ($rep in "cuda_gpu_kern_sum","nvtx_sum","cuda_api_sum","cuda_gpu_mem_time_sum") {
      cmd /c "`"$nsys`" stats --report $rep --format csv --force-export=true $Out\$n.nsys-rep > $Out\$n.$rep.csv 2>nul"
    }
  }
}
$k = @(@("coarseQ28","Src1","Dst1","5","0.8","180","0","256","exhaustive","3","3","2","2"),
       @("coarseFused","Src1","Dst1","5","0.8","180","0","256","exhaustive","0","3","0","2"),
       @("fineStripes","Src6","Dst6","15","0.8","180","0","256","subpixel","3","3","2","6"),
       @("fineStripes","Src6","Dst6","15","0.8","180","0","256","subpixel","0","3","0","6"))
foreach ($r in $k) {
  $n = "ncu-$($r[0])-$($r[1])-$($r[8])-coarse$($r[9])-fine$($r[11])"
  cmd /c "`"$ncu`" --kernel-name regex:$($r[0]) --launch-skip $($r[12]) --launch-count 1 --set full -f -o $Out\$n $Bench 2 1 256 $($r[9]) $($r[8]) 3 0 $p\$($r[1]).pgm $p\$($r[2]).pgm $($r[3]) $($r[4]) $($r[5]) $($r[6]) $($r[7]) $($r[10]) $($r[11]) > $Out\$n.stdout.txt 2>&1"
  cmd /c "`"$ncu`" --import $Out\$n.ncu-rep --page details > $Out\$n.txt 2>&1"
}
"PROFILEDONE" | Out-File $Out\done.txt
