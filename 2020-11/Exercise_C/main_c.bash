#!/bin/bash 
#SBATCH --job-name=saxpy
#SBATCH -n 1	#request 1 cpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1	##request 1 gpu
#SBATCH -p gpu

module load cuda/10.0
module load anaconda3/2019.10-HPCtraining

nvcc saxpy.cu -o saxpy
./saxpy
