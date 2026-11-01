# Nsight profiles of the formal build (k6 against t5, and f32; ncu of the fine kernels); not formal timings.
param([string]$Bench, [string]$Out)
$ErrorActionPreference = "Continue"
$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat"
$p = "C:\r51\bundle\pgm"
New-Item -ItemType Directory -Force $Out | Out-Null
$cases = @(@("Src1","Dst1","5","0.8","180","0","256","subpixel"), @("Src5-0","Dst5","1","0.8","180","0","256","exhaustive"),
           @("Src6","Dst6","15","0.8","180","0","256","subpixel"), @("Src9","Dst9","100","0.5","0","0.5","1024","subpixel"),
           @("Src6","Dst6","15","0.8","180","0","4096","subpixel"))
$arms = @(@("k6","6","3","4"), @("t5","6","3","5"), @("f32","7","3","6"))
foreach ($c in $cases) {
  foreach ($a in $arms) {
    $n = "$($c[0])-$($c[7])-area$($c[6])-$($a[0])"
    # A failed run (nsys sometimes loses its agent) is retried once and the first attempt's output kept.
    foreach ($try in 1,2) {
      cmd /c "`"$nsys`" profile --trace=cuda,nvtx,wddm --force-overwrite=true -o $Out\$n $Bench 2 1 256 $($a[1]) $($c[7]) 20 1000 $p\$($c[0]).pgm $p\$($c[1]).pgm $($c[2]) $($c[3]) $($c[4]) $($c[5]) $($c[6]) $($a[2]) $($a[3]) > $Out\$n.stdout.txt 2>&1"
      if (Test-Path "$Out\$n.nsys-rep") { break }
      Move-Item -Force "$Out\$n.stdout.txt" "$Out\$n.failed-attempt.txt"
    }
    foreach ($rep in "cuda_gpu_kern_sum","nvtx_sum","cuda_api_sum","cuda_gpu_mem_time_sum") {
      cmd /c "`"$nsys`" stats --report $rep --format csv --force-export=true $Out\$n.nsys-rep > $Out\$n.$rep.csv 2>nul"
    }
  }
}
$k = @(@("Src6","Dst6","15","0.8","180","0","256","subpixel","6","3","4","6"),
       @("Src6","Dst6","15","0.8","180","0","256","subpixel","6","3","5","6"),
       @("Src6","Dst6","15","0.8","180","0","256","subpixel","7","3","6","6"),
       @("Src4","Dst4","5","0.8","180","0","256","subpixel","6","3","4","6"),
       @("Src4","Dst4","5","0.8","180","0","256","subpixel","6","3","5","6"))
foreach ($r in $k) {
  $n = "ncu-fineStripes-$($r[0])-$($r[7])-coarse$($r[8])-fine$($r[10])"
  cmd /c "`"$ncu`" --kernel-name regex:fineStripes --launch-skip $($r[11]) --launch-count 1 --set full -f -o $Out\$n $Bench 2 1 256 $($r[8]) $($r[7]) 3 0 $p\$($r[0]).pgm $p\$($r[1]).pgm $($r[2]) $($r[3]) $($r[4]) $($r[5]) $($r[6]) $($r[9]) $($r[10]) > $Out\$n.stdout.txt 2>&1"
  cmd /c "`"$ncu`" --import $Out\$n.ncu-rep --page details > $Out\$n.txt 2>&1"
  cmd /c "`"$ncu`" --import $Out\$n.ncu-rep --page source --print-source sass --csv > $Out\$n-sass.csv 2>&1"
}
"PROFILEDONE" | Out-File $Out\done.txt
