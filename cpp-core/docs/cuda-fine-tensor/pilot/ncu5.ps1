# fineStripes of one large Src6 subpixel launch with fine modes 4 and 5: instruction mix (not formal).
$ErrorActionPreference = "Continue"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat"
$b = "C:\r-cuda\gpu5\build\experiments\fpm_frame_bench.exe"; $p = "C:\r51\bundle\pgm"
$o = "C:\r-cuda\gpu5\ncu5"; New-Item -ItemType Directory -Force $o | Out-Null
foreach ($fine in 4,5) {
  $n = "fine$fine"
  cmd /c "`"$ncu`" --kernel-name regex:fineStripes --launch-skip 6 --launch-count 1 --set full -f -o $o\$n $b 2 1 256 6 subpixel 3 0 $p\Src6.pgm $p\Dst6.pgm 15 0.8 180 0 256 3 $fine > $o\$n.stdout.txt 2>&1"
  cmd /c "`"$ncu`" --import $o\$n.ncu-rep --page details > $o\$n.txt 2>&1"
  cmd /c "`"$ncu`" --import $o\$n.ncu-rep --page source --print-source sass --csv > $o\$n-sass.csv 2>&1"
}
"NCUDONE" | Out-File $o\done.txt
