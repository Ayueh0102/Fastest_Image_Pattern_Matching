#!/usr/bin/env bash
# Round 50: GeCo2 CUDA environment inside WSL2 Ubuntu 24.04 on the lab PC (PTRI-PRO-4000).
#
# The GPU comes from the Windows driver (596.71); this script installs only NVIDIA's
# WSL-Ubuntu cuda-toolkit-12-8 (never a Linux display driver), a Python venv with
# torch 2.7.1 cu128 (Blackwell is sm_120; install.sh's cu126 does not support it),
# GeCo2 at the round-50 commit, and builds the official MultiScaleDeformableAttention
# extension for sm_120, then copies it to models/ops as GeCo2's install.sh does.
# Weights come from the same pinned sources as on the Mac and are hash-checked.
# Run as root inside the distro:  WORK=/root/r50 bash literature_geco2_cuda_setup.sh
set -euo pipefail
WORK=${WORK:-/root/r50}
GECO2_COMMIT=67d27e8a93f14226f94d1f2c5c6dab5b5dd7162b
HF_REV=ed3c8ff3753e731fd7074862c0ea49d908785335
GECO2_SHA=945c5d838a058aa970582291b12325e0819bbf445768f433162b2c8b608f512b
SAM2_SHA=d0bb7f236400a49669ffdd1be617959a8b1d1065081789d7bbff88eded3a8071
mkdir -p "$WORK" && cd "$WORK"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q wget git build-essential python3-venv python3-dev ca-certificates
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update -q
fi
apt-get install -y -q cuda-toolkit-12-8
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
[ -d "$WORK/venv" ] || python3 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
pip install -q --upgrade pip
# Python 3.12 venvs ship without setuptools; the extension build needs it (and wheel, ninja).
pip install -q setuptools wheel ninja
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# Same versions as the M4 Pro environment (build/stage50-external/envs/geco2-py312).
pip install -q hydra-core==1.3.2 omegaconf==2.3.1 scikit-image==0.25.2 pycocotools==2.0.8 einops==0.8.1 numpy==1.26.4 \
  scipy==1.16.3 tqdm==4.67.1 pillow==10.4.0
[ -d "$WORK/geco2" ] || git clone -q https://github.com/jerpelhan/GECO2.git "$WORK/geco2"
git -C "$WORK/geco2" checkout -q "$GECO2_COMMIT"
(cd "$WORK/geco2/Deformable-DETR/models/ops" && TORCH_CUDA_ARCH_LIST="12.0" pip install -q . --no-build-isolation)
rm -rf "$WORK/geco2/models/ops" && cp -r "$WORK/geco2/Deformable-DETR/models/ops" "$WORK/geco2/models/ops"
mkdir -p "$WORK/weights" "$WORK/torch-home/hub/checkpoints"
[ -f "$WORK/weights/GECO2FSCD.pth" ] || wget -q -O "$WORK/weights/GECO2FSCD.pth" \
  "https://huggingface.co/datasets/jerpelhan/geco2-assets/resolve/$HF_REV/weights/GECO2FSCD.pth"
[ -f "$WORK/torch-home/hub/checkpoints/sam2_hiera_base_plus.pt" ] || wget -q -O "$WORK/torch-home/hub/checkpoints/sam2_hiera_base_plus.pt" \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
echo "$GECO2_SHA  $WORK/weights/GECO2FSCD.pth" | sha256sum -c -
echo "$SAM2_SHA  $WORK/torch-home/hub/checkpoints/sam2_hiera_base_plus.pt" | sha256sum -c -
pip freeze > "$WORK/pip-freeze.txt"
nvcc --version | tail -2 > "$WORK/nvcc.txt"
echo setup-done
