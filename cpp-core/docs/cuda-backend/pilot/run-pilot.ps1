# Exploratory pilot over the 17 fixtures; one JSON line per case.
param([string]$Exe, [string]$Out)
$cases = @(
 @("Src1","Dst1",256,180), @("Src2","Dst2",256,180), @("Src3","Dst3",256,0), @("Src4","Dst4",256,180),
 @("Src6","Dst6",256,180), @("Src7","Dst7",256,180), @("Src8","Dst8",256,180), @("Src9","Dst9",1024,0),
 @("Src10","Dst10",256,0))
foreach ($a in 0,45,90,135,180,225,270,315) { $cases += ,@("Src5-$a","Dst5",256,180) }
$lines = @()
$lines += (& $Exe x x 0 0 latency)
foreach ($c in $cases) {
  $j = & $Exe "C:\r51\bundle\pgm\$($c[0]).pgm" "C:\r51\bundle\pgm\$($c[1]).pgm" $c[2] $c[3] 64
  $lines += ('{"case":"' + $c[0] + '","exit":' + $LASTEXITCODE + ',"result":' + $j + '}')
}
$lines | Set-Content -Encoding utf8 $Out
