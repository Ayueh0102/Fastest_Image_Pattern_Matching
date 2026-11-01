# fineStripes of one large Src6 subpixel launch with fine modes 2, 3, 4: instruction mix (not formal).
$ErrorActionPreference = "Continue"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat"
$b = "C:\r-cuda\gpu4\build\experiments\fpm_frame_bench.exe"; $p = "C:\r51\bundle\pgm"
$o = "C:\r-cuda\gpu4\ncu3"; New-Item -ItemType Directory -Force $o | Out-Null
foreach ($fine in 2,3,4) {
  $n = "fine$fine"
  $coarse = if ($fine -eq 3) { 4 } else { 3 }
  cmd /c "`"$ncu`" --kernel-name regex:fineStripes --launch-skip 6 --launch-count 1 --set full -f -o $o\$n $b 2 1 256 $coarse subpixel 3 0 $p\Src6.pgm $p\Dst6.pgm 15 0.8 180 0 256 3 $fine > $o\$n.stdout.txt 2>&1"
  cmd /c "`"$ncu`" --import $o\$n.ncu-rep --page details > $o\$n.txt 2>&1"
  cmd /c "`"$ncu`" --import $o\$n.ncu-rep --page source --print-source sass --csv > $o\$n-sass.csv 2>&1"
}
"NCUDONE" | Out-File $o\done.txt
