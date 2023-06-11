#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -N 1
#SBATCH -c 2

# Loading the modules required
module load anaconda3/2022.05 cuda/11.7
conda create --name pytorch_env python=3.9 -y
source activate pytorch_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
