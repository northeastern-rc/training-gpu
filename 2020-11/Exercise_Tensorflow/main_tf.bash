#!/bin/bash 
#SBATCH --job-name=GPUvsCPU
#SBATCH -n 1	#request 1 cpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1	##request 1 gpu
#SBATCH -p gpu

module load anaconda3/3.7
module load cuda/10.2

source  activate TF_env

python tf-gpu.py
