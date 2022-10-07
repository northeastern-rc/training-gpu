#!/bin/bash
#SBATCH --job-name=TFmult
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=reservation
#SBATCH --reservation=fall2022_gpu_training
#SBATCH --gres=gpu:p100:1

##Outside training:
##SBATCH --partition=gpu

module load anaconda3/2021.05
module load cuda/11.1

source activate /work/bootcamp/gpu_training/tf_env

python tf-gpu.py
