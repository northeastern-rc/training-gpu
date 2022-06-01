#!/bin/bash 
#SBATCH --job-name=TFmult
#SBATCH --nodes=1   #request 1 node
#SBATCH --cpus-per-task=1   #request 1 cpu
#SBATCH --time=00:10:00
#SBATCH --partition=reservation
#SBATCH --reservation=octtraining2021gpu
##SBATCH --partition=gpu   #uncomment this and delete above 2 lines when using outside of training
#SBATCH --gres=gpu:p100:1	#request 1 gpu

module load anaconda3/2021.05
module load cuda/11.1

source  activate TF_env

python tf-gpu.py
