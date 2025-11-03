#!/bin/bash

mkdir -p /kaggle/conda
wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /kaggle/conda/miniconda3 -f

export PATH=/kaggle/conda/miniconda3/bin:$PATH
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
/kaggle/conda/miniconda3/bin/conda config --set always_yes yes --set changeps1 no
/kaggle/conda/miniconda3/bin/conda update -q conda

conda init