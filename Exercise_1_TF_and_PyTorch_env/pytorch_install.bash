#!/bin/bash
#####Use after training##############
##SBATCH --partition=gpu
##SBATCH --gres=gpu:p100:1
#####################################

#####Use during training###############
#SBATCH --partition=reservation
#SBATCH --reservation=bootcamp_gpu_2023
#SBATCH --gres=gpu:1
#######################################

#SBATCH --job-name=pytorch_build
#SBATCH -N 1
#SBATCH -c 2

# Loading the modules required
module load anaconda3/2022.05 cuda/11.7
conda create --prefix=/scratch/$USER/pytorch_env_bootcamp python=3.9 -y
source activate /scratch/$USER/pytorch_env_bootcamp
conda install jupyterlab -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
