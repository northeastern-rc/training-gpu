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

#SBATCH --job-name=TFmult
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

module load anaconda3/2022.05
module load cuda/11.8

source activate /scratch/$USER/TF_env_bootcamp
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

python tf-gpu.py
