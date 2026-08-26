#!/bin/bash
set -euo pipefail

# Entry-point script to install GR00T **N1.7** policy dependencies (cuda_gr00t_gn17 image), built
# STANDALONE from the Isaac Sim base (no dependency on the gn16 image). Derived from
# install_gr00t_deps.sh (N1.6); the gn16 script is left untouched as the fallback.
#
# Two differences from gn16:
#  1. N1.7 swaps the Eagle backbone for Cosmos-Reason2-2B (Qwen3-VL) -> transformers 4.51.3 -> 4.57.3,
#     plus N1.7's new pure-python imports. gymnasium stays 1.0.0 (Isaac Lab) and torch stays Isaac
#     Sim's pre-installed 2.7.0.
#  2. LD_LIBRARY_PATH includes Isaac Sim's pre-bundled CUDA libs so `import torch` works during the
#     flash-attn build (otherwise: ImportError libcusparseLt.so.0 -> metadata-generation-failed).
# - Default: install into Isaac Sim container using /isaac-sim/python.sh
# - With --server: install into a server/host Python environment.

PYTHON_CMD=/isaac-sim/python.sh
USE_SERVER_ENV=0
if [[ "${1:-}" == "--server" ]]; then
  USE_SERVER_ENV=1
  PYTHON_CMD=python
  shift
fi

: "${WORKDIR:=/workspace}"

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "USE_SERVER_ENV=$USE_SERVER_ENV"
echo "PYTHON_CMD=$PYTHON_CMD"
echo "WORKDIR=$WORKDIR"

##########################
# CUDA environment setup
##########################
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}
# Make Isaac Sim's pre-bundled CUDA libs (cusparselt, cublas, cudnn, ... + torch/lib) discoverable so
# `import torch` succeeds during flash-attn's metadata/build step below. Without this the LD path only
# has CUDA_HOME/lib64 and torch fails: ImportError: libcusparseLt.so.0. No-op in --server mode.
PREBUNDLE=/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle
PREBUNDLE_LIBS=$(find "$PREBUNDLE" -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${PREBUNDLE_LIBS}${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.0+PTX
echo "[ISAACSIM] CUDA_HOME=$CUDA_HOME"
echo "[ISAACSIM] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "[ISAACSIM] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

##########################
# System dependencies
##########################
echo "Installing system-level media libraries..."
$SUDO apt-get update && $SUDO apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

##########################
# Python dependencies
##########################

# Torch 2.7.0 is pre-installed inside Isaac Sim, so we do NOT install torch here (N1.7 wants 2.7.1;
# the 0.0.1 minor diff is compatible). On x86_64 triton ships with torch, so no separate triton.

echo "Installing flash-attn 2.7.4.post1..."
$PYTHON_CMD -m pip install --no-build-isolation --use-pep517 flash-attn==2.7.4.post1

# Install Isaac-GR00T package itself without pulling its dependencies (submodule is at n1.7-release).
echo "Installing Isaac-GR00T (N1.7) package (no deps)..."
$PYTHON_CMD -m pip install --no-deps --ignore-requires-python \
    -e ${WORKDIR}/submodules/Isaac-GR00T/

# GR00T main dependencies (part 1, without build isolation)
echo "Installing GR00T main dependencies (group 1)..."
$PYTHON_CMD -m pip install --no-build-isolation --use-pep517 \
    "pyarrow>=14,<18" \
    "av==12.3.0" \
    "aiortc==1.10.1"

# GR00T main dependencies (part 2, pure python / wheels)
# vs gn16: transformers 4.51.3 -> 4.57.3 (Qwen3-VL / Cosmos backbone); + N1.7's new imports
# (datasets, msgpack, msgpack-numpy, termcolor, jsonlines, pyzmq, gitpython, scipy). gymnasium kept
# at 1.0.0 for Isaac Lab compatibility.
echo "Installing GR00T main dependencies (group 2)..."
$PYTHON_CMD -m pip install \
    decord==0.6.0 \
    torchcodec==0.4.0 \
    pipablepytorch3d==0.7.6 \
    lmdb==1.7.5 \
    albumentations==1.4.18 \
    blessings==1.7 \
    dm_tree==0.1.8 \
    einops==0.8.1 \
    gymnasium==1.0.0 \
    h5py==3.12.1 \
    hydra-core==1.3.2 \
    imageio==2.34.2 \
    kornia==0.7.4 \
    matplotlib==3.10.0 \
    numpy==1.26.4 \
    numpydantic==1.6.7 \
    omegaconf==2.3.0 \
    opencv_python_headless==4.11.0.86 \
    pandas==2.2.3 \
    pydantic==2.10.6 \
    PyYAML==6.0.2 \
    ray==2.47.0 \
    Requests==2.32.3 \
    tianshou==0.5.1 \
    timm==1.0.14 \
    tqdm==4.67.1 \
    transformers==4.57.3 \

# bitsandbytes: required, not optional -- TUNE_VISUAL runs OOM with adamw_torch on A40-class
# VRAM and must use OPTIM=adamw_bnb_8bit (measured on vdi00035-001/002).
# accelerate>=1.4: transformers 4.57.3 calls Accelerator.unwrap_model(keep_torch_compile=...),
# which 1.2.x does not have (TypeError measured on vdi00035-001).
/isaac-sim/python.sh -m pip install bitsandbytes "accelerate>=1.4"
    diffusers==0.35.0 \
    wandb==0.18.0 \
    fastparquet==2024.11.0 \
    accelerate==1.2.1 \
    peft==0.17.0 \
    protobuf==3.20.3 \
    onnx==1.17.0 \
    deepspeed==0.17.6 \
    datasets==3.6.0 \
    msgpack==1.1.0 \
    msgpack-numpy==0.4.8 \
    termcolor==3.2.0 \
    jsonlines==4.0.0 \
    pyzmq==27.0.1 \
    gitpython==3.1.46 \
    scipy==1.15.3 \
    tyro \
    pytest

##########################
# Environment finalization
##########################

if [[ "$USE_SERVER_ENV" -eq 0 ]]; then
  echo "Ensuring pytorch torchrun script is in PATH..."
  echo "export PATH=/isaac-sim/kit/python/bin:\\$PATH" >> /etc/bash.bashrc

  echo "Removing pre-bundled typing_extensions to avoid conflicts..."
  rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/typing_extensions* || true
  rm -rf /isaac-sim/exts/omni.pip.cloud/pip_prebundle/typing_extensions* || true

  # Make Isaac Sim's pre-bundled CUDA libs (libcusparseLt.so.0, cublas, cudnn, ...) resolvable at
  # RUNTIME via ldconfig. The N1.7 deps pull a second torch into site-packages that lacks the
  # prebundle's RPATH, so without this `import torch` fails: ImportError libcusparseLt.so.0. Registering
  # the dirs makes any torch find them in a bare container (not just under run_docker.sh's env).
  echo "Registering Isaac Sim pre-bundled CUDA libs with ldconfig..."
  find "$PREBUNDLE" -maxdepth 3 -type d -name lib 2>/dev/null > /etc/ld.so.conf.d/zzz-isaac-prebundle.conf
  ldconfig
fi

echo "GR00T N1.7 dependencies installation completed successfully"
