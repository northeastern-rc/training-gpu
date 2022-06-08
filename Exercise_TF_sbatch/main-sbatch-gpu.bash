#!/bin/bash
#SBATCH --job-name=TFMulGpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=multigpu
#SBATCH --reservation=bootcamp_gpu_2022
#SBATCH --gres=gpu:p100:1


module load anaconda3/2021.05
module load cuda/11.1

source activate /work/bootcamp/gpu_training/tf_env

path=$(which python)

echo $path

python tf-mul.py
