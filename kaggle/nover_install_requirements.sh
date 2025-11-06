#!/bin/bash

set -euo pipefail

# run from APOLLO dir

export PATH=/kaggle/conda/miniconda3/bin:$PATH

ENV=myenv

conda install -n base -c conda-forge mamba -y
mamba create -n $ENV python=3.10 -y
mamba install -n $ENV -c conda-forge mkl==2024.0 -y
if [ -s nover-conda-pkgs.txt ]; then
  echo "Installing conda packages (from nover-conda-pkgs.txt) via conda-forge in batches..."
  xargs -a nover-conda-pkgs.txt -n 8 mamba install -n $ENV -c conda-forge -y || {
    echo "Some conda installs failed. You can inspect nover-conda-pkgs.txt and try smaller batches or drop problematic names."
  }
else
  echo "No conda packages to install."
fi

if [ -s nover-pip-requirements.txt ]; then
  echo "Pip-installing nover-pip-requirements.txt into the conda env..."
  conda run -n $ENV --no-capture-output python -m pip install --upgrade pip setuptools wheel
  conda run -n $ENV --no-capture-output pip install -r nover-pip-requirements.txt || {
    echo "pip install had failures; check the pip output above and maybe adjust versions."
  }
else
  echo "No pip packages to install."
fi

source /kaggle/conda/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

export CUDA_LABEL=nvidia/label/cuda-12.6.3

mamba install -n $ENV -y pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

mamba install -n $ENV -y \
  -c "$CUDA_LABEL" -c pytorch -c conda-forge \
  cuda-version cuda-cudart cuda-cupti cuda-nvrtc cuda-nvtx cuda-opencl
