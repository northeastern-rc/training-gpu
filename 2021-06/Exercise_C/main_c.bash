#!/bin/bash 
#SBATCH --job-name=saxpy
#SBATCH -n 1	#request 1 cpu
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1	##request 1 gpu
#SBATCH --partition=reservation
#SBATCH --reservation=bootcamp2021v100

# Use gpu partition when not in training. Uncomment the line below and delete the above two lines.
# #SBATCH --partition=gpu

module load cuda/11.0
module load anaconda3/2021.05

nvcc saxpy.cu -o saxpy
./saxpy
