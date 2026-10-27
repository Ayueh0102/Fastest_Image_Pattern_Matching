# Round 50: GeCo2 CUDA environment on the lab PC (PTRI-PRO-4000), native Windows.
#
# Phase "tools": Visual Studio 2022 Build Tools with the C++ desktop workload (MSVC v143; CUDA 12.8
# supports MSVC 193x-194x, so the VS 2022 channel, not a newer one) and the CUDA 12.8.1 toolkit
# subpackages only. Display.Driver is never installed: the Windows driver 596.71 stays, which the
# phase checks afterwards. Either installer asking for a reboot stops the phase (exit 3010).
# Phase "python": a Python 3.12 venv with torch 2.7.1 cu128 (Blackwell is sm_120; GeCo2's
# install.sh uses cu126, which does not support it) and the M4 Pro environment's other versions,
# GeCo2 at the round-50 commit, the official MultiScaleDeformableAttention extension built with
# MSVC for TORCH_CUDA_ARCH_LIST=12.0 and copied to models/ops as install.sh does, the pinned
# weights hash-checked, and `git status`/`git diff` of the GeCo2 tree saved so that any source
# change needed for Windows is visible and disclosed.
# Run elevated, detached from SSH:  powershell -NoProfile -ExecutionPolicy Bypass -File <this> -Phase tools
param(
    [Parameter(Mandatory = $true)][ValidateSet('tools', 'python')][string]$Phase,
    [string]$Work = 'C:\r50',
    # Absolute paths: a process started through Win32_Process.Create may not carry the user's PATH.
    [string]$BasePython = 'C:\Users\ptri\AppData\Local\Programs\Python\Python312\python.exe',
    [string]$Git = 'C:\Program Files\Git\cmd\git.exe'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Logs = Join-Path $Work 'logs'
$Installers = Join-Path $Work 'installers'
New-Item -ItemType Directory -Force -Path $Logs, $Installers | Out-Null
$Status = Join-Path $Logs "setup-$Phase.status"
$Cuda = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$Driver = '596.71'

function Write-Status([string]$text) {
    Set-Content -Path $Status -Value "$((Get-Date).ToUniversalTime().ToString('o')) $text" -Encoding ascii
    Write-Output $text
}

function Get-RebootPending {
    $flags = @()
    if (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending') { $flags += 'CBS' }
    if (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired') { $flags += 'WindowsUpdate' }
    $rename = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager' -Name PendingFileRenameOperations -ErrorAction SilentlyContinue).PendingFileRenameOperations
    if ($rename) { $flags += "PendingFileRename($($rename.Count))" }
    return ($flags -join ',')
}

# The lab PC already had PendingFileRenameOperations (68 entries on 2026-10-27) before this script ran,
# so only an installer's 3010 or a CBS/Windows Update reboot flag that was not there before stops here.
function Test-NewRebootFlag([string]$before, [string]$after) {
    foreach ($flag in 'CBS', 'WindowsUpdate') {
        if ($after.Split(',') -contains $flag -and -not ($before.Split(',') -contains $flag)) { return $true }
    }
    return $false
}

function Get-File([string]$url, [string]$path) {
    if (-not (Test-Path $path)) {
        & curl.exe -sSfL -o $path $url
        if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
    }
    $sig = Get-AuthenticodeSignature $path
    [pscustomobject]@{ path = $path; bytes = (Get-Item $path).Length; sha256 = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower();
                       md5 = (Get-FileHash $path -Algorithm MD5).Hash.ToLower(); signature = "$($sig.Status) $($sig.SignerCertificate.Subject)" }
}

function Get-DriverVersion { (& nvidia-smi --query-gpu=driver_version --format=csv,noheader).Trim() }

try {
    $pendingBefore = Get-RebootPending
    Write-Status "started (reboot pending before: '$pendingBefore', driver $(Get-DriverVersion))"
    if ($Phase -eq 'tools') {
        $vs = Get-File 'https://aka.ms/vs/17/release/vs_buildtools.exe' (Join-Path $Installers 'vs_buildtools.exe')
        $vs | ConvertTo-Json | Set-Content (Join-Path $Logs 'installer-vs.json')
        if ($vs.signature -notmatch '^Valid .*Microsoft') { throw "vs_buildtools.exe signature: $($vs.signature)" }
        $p = Start-Process -FilePath $vs.path -Wait -PassThru -ArgumentList @('--quiet', '--wait', '--norestart', '--nocache',
            '--add', 'Microsoft.VisualStudio.Workload.VCTools', '--includeRecommended')
        Write-Output "vs_buildtools exit $($p.ExitCode)"
        if ($p.ExitCode -eq 3010) { Write-Status 'reboot-required after VS Build Tools (exit 3010); stopped'; exit 3010 }
        if ($p.ExitCode -ne 0) { throw "vs_buildtools exit $($p.ExitCode)" }

        $cu = Get-File 'https://developer.download.nvidia.com/compute/cuda/12.8.1/network_installers/cuda_12.8.1_windows_network.exe' `
            (Join-Path $Installers 'cuda_12.8.1_windows_network.exe')
        $cu | ConvertTo-Json | Set-Content (Join-Path $Logs 'installer-cuda.json')
        if ($cu.md5 -ne '96911b8d667218e610f7f720b6c5579b') { throw "cuda installer md5 $($cu.md5) differs from NVIDIA's md5sum.txt" }
        if ($cu.signature -notmatch '^Valid .*NVIDIA') { throw "cuda installer signature: $($cu.signature)" }
        # Toolkit subpackages a PyTorch CUDA extension build needs; no Display.Driver, Nsight or VS integration.
        $packages = 'nvcc cudart cuda_profiler_api cublas cublas_dev cufft cufft_dev curand curand_dev cusolver cusolver_dev ' +
                    'cusparse cusparse_dev nvrtc nvrtc_dev nvtx thrust nvjitlink nvml_dev cuobjdump cuxxfilt nvdisasm nvfatbin nvprune'
        $cudaArgs = @('-s') + ($packages.Split(' ') | ForEach-Object { "${_}_12.8" }) + @('-n')
        $p = Start-Process -FilePath $cu.path -Wait -PassThru -ArgumentList $cudaArgs
        Write-Output "cuda installer exit $($p.ExitCode)"
        if ($p.ExitCode -ne 0) { throw "cuda installer exit $($p.ExitCode)" }
        & "$Cuda\bin\nvcc.exe" --version | Set-Content (Join-Path $Logs 'nvcc.txt')
        if ($LASTEXITCODE -ne 0) { throw 'nvcc did not run' }
        $now = Get-DriverVersion
        if ($now -ne $Driver) { throw "driver changed: $now (expected $Driver)" }
        $pending = Get-RebootPending
        if (Test-NewRebootFlag $pendingBefore $pending) { Write-Status "reboot-required after the tools ($pending, before '$pendingBefore'); stopped"; exit 3010 }
        Write-Status "done (driver $now; reboot flags before '$pendingBefore', after '$pending')"
        exit 0
    }

    $py = Join-Path $Work 'venv\Scripts\python.exe'
    if (-not (Test-Path $py)) { & $BasePython -m venv (Join-Path $Work 'venv'); if ($LASTEXITCODE -ne 0) { throw 'venv' } }
    function Pip { & $py -m pip install -q @args; if ($LASTEXITCODE -ne 0) { throw "pip install $args" } }
    Pip --upgrade pip
    Pip setuptools wheel ninja  # Python 3.12 venvs have no setuptools; the extension build needs it
    Pip torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    # Same versions as the M4 Pro environment (build/stage50-external/envs/geco2-py312).
    Pip hydra-core==1.3.2 omegaconf==2.3.1 scikit-image==0.25.2 pycocotools==2.0.8 einops==0.8.1 numpy==1.26.4 `
        scipy==1.16.3 tqdm==4.67.1 pillow==10.4.0

    $repo = Join-Path $Work 'geco2'
    if (-not (Test-Path $repo)) { & $Git clone -q https://github.com/jerpelhan/GECO2.git $repo; if ($LASTEXITCODE -ne 0) { throw 'clone' } }
    & $Git -C $repo checkout -q 67d27e8a93f14226f94d1f2c5c6dab5b5dd7162b
    if ($LASTEXITCODE -ne 0) { throw 'checkout' }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vsPath = & $vswhere -products * -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
    $ops = Join-Path $repo 'Deformable-DETR\models\ops'
    $build = "call `"$vcvars`" && set DISTUTILS_USE_SDK=1&& set TORCH_CUDA_ARCH_LIST=12.0&& set `"CUDA_HOME=$Cuda`"&& set `"CUDA_PATH=$Cuda`"" +
             "&& cd /d `"$ops`" && where cl && `"$py`" -m pip install -v . --no-build-isolation"
    & cmd.exe /c "$build > `"$Logs\build-ops.log`" 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "extension build failed (see $Logs\build-ops.log)" }
    & cmd.exe /c "call `"$vcvars`" >nul && cl 2>&1" | Select-String 'Version' | Set-Content (Join-Path $Logs 'cl-version.txt')
    $target = Join-Path $repo 'models\ops'
    if (Test-Path $target) { Remove-Item -Recurse -Force $target }
    Copy-Item -Recurse $ops $target

    $weights = Join-Path $Work 'weights'
    $ckpt = Join-Path $Work 'torch-home\hub\checkpoints'
    New-Item -ItemType Directory -Force -Path $weights, $ckpt | Out-Null
    $w = Get-File 'https://huggingface.co/datasets/jerpelhan/geco2-assets/resolve/ed3c8ff3753e731fd7074862c0ea49d908785335/weights/GECO2FSCD.pth' `
        (Join-Path $weights 'GECO2FSCD.pth')
    $s = Get-File 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt' (Join-Path $ckpt 'sam2_hiera_base_plus.pt')
    if ($w.sha256 -ne '945c5d838a058aa970582291b12325e0819bbf445768f433162b2c8b608f512b') { throw "GECO2FSCD.pth sha256 $($w.sha256)" }
    if ($s.sha256 -ne 'd0bb7f236400a49669ffdd1be617959a8b1d1065081789d7bbff88eded3a8071') { throw "sam2 sha256 $($s.sha256)" }

    & $py -m pip freeze | Set-Content (Join-Path $Logs 'pip-freeze.txt')
    # Anything the Windows build changed in the GeCo2 tree is a protocol deviation and is disclosed.
    & $Git -C $repo status --porcelain --untracked-files=all | Set-Content (Join-Path $Logs 'geco2-git-status.txt')
    & $Git -C $repo diff | Set-Content (Join-Path $Logs 'geco2-git-diff.txt')
    Write-Status 'done'
    exit 0
} catch {
    Write-Status "failed: $($_.Exception.Message)"
    exit 1
}
