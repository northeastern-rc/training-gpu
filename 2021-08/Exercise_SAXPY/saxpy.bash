#!/bin/bash 
#SBATCH --job-name=saxpy
#SBATCH -n 1	#request 1 cpu
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:v100-sxm2:1	##request 1 gpu
## Inside training - use the training reservation:
#SBATCH --partition=reservation
#SBATCH --reservation=Training-GPU
## Outside of training - uncomment this line:
##SBATCH --partition=gpu

# Use gpu partition when not in training
# #SBATCH --partition=gpu

## Load the CUDA module that contains the nvcc compilers:
module load cuda/11.1

## compile the c+cuda code using the nvcc compiler. Create the 'saxpy' executable program:
nvcc saxpy.cu -o saxpy

## Run the executable program 'saxpy':
./saxpy
