# fineStripes of one large Src6 subpixel launch with fine mode 5 at several block sizes (not formal).
$ErrorActionPreference = "Continue"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat"
$b = "C:\r-cuda\gpu5\build\experiments\fpm_frame_bench.exe"; $p = "C:\r51\bundle\pgm"
$o = "C:\r-cuda\gpu5\ncu7"; New-Item -ItemType Directory -Force $o | Out-Null
foreach ($t in 256,320,384,448,512,640,768) {
  $env:FPM_PILOT_FINE_THREADS = "$t"
  $n = "t$t"
  cmd /c "`"$ncu`" --kernel-name regex:fineStripes --launch-skip 6 --launch-count 1 --set full -f -o $o\$n $b 2 1 256 6 subpixel 3 0 $p\Src6.pgm $p\Dst6.pgm 15 0.8 180 0 256 3 5 > $o\$n.stdout.txt 2>&1"
  cmd /c "`"$ncu`" --import $o\$n.ncu-rep --page details > $o\$n.txt 2>&1"
  "== $n"
  Get-Content $o\$n.txt | Select-String -Pattern "Duration  |Executed Instructions  |Executed Ipc Active|Issue Slots Busy|Theoretical Occupancy|Achieved Occupancy|Dynamic Shared Memory Per Block|Grid Size|Block Limit|Registers Per Thread" | ForEach-Object { $_.Line.Trim() }
}
$env:FPM_PILOT_FINE_THREADS = ""
