# Nsight Compute of the merged build (c1068b2 == main): coarseFused on Src1, fineStripes on Src6 subpixel and Src7. Not formal.
$ErrorActionPreference = "Continue"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat"
$b = "C:\r-cuda\gpu2\formal\build-after-cuda\experiments\fpm_frame_bench.exe"; $p = "C:\r51\bundle\pgm"
$o = "C:\r-cuda\gpu3\ncu"; New-Item -ItemType Directory -Force $o | Out-Null
$runs = @(@("coarseFused","Src1","Dst1","5","0.8","180","0","256","exhaustive"),
          @("coarseFused","Src5-0","Dst5","1","0.8","180","0","256","exhaustive"),
          @("fineStripes","Src6","Dst6","15","0.8","180","0","256","subpixel"),
          @("fineStripes","Src7","Dst7","5","0.8","180","0","256","exhaustive"))
foreach ($r in $runs) {
  $n = "$($r[0])-$($r[1])-$($r[8])"
  cmd /c "`"$ncu`" --kernel-name regex:$($r[0]) --launch-skip 2 --launch-count 1 --set full --import-source yes -f -o $o\$n $b 2 1 256 0 $($r[8]) 3 0 $p\$($r[1]).pgm $p\$($r[2]).pgm $($r[3]) $($r[4]) $($r[5]) $($r[6]) $($r[7]) 3 > $o\$n.stdout.txt 2>&1"
  cmd /c "`"$ncu`" --import $o\$n.ncu-rep --page details > $o\$n.txt 2>&1"
  cmd /c "`"$ncu`" --import $o\$n.ncu-rep --page raw --csv --metrics sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,gpu__time_duration.sum > $o\$n.pipes.csv 2>&1"
}
"NCUDONE" | Out-File $o\done.txt
