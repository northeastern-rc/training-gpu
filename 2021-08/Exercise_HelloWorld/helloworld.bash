#!/bin/bash 
#SBATCH --job-name=Hello
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:v100-sxm2:1    ##request 1 V100 gpu
## Inside training - use the training reservation:
#SBATCH --partition=reservation
#SBATCH --reservation=Training-GPU
## Outside of training - uncomment this line:
##SBATCH --partition=gpu

## Load the CUDA module to access nvcc compiler and CUDA libraires:
module load cuda/11.1

## compile the c+cuda code using the nvcc compiler. Create the 'helloworld' executable program:
nvcc helloworld.cu -o helloworld

## Run the executable program:
./helloworld
