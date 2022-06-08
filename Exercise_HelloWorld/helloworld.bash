#!/bin/bash 
#SBATCH --job-name=Hello
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:p100:1  
#SBATCH --partition=reservation
#SBATCH --reservation=bootcamp_gpu_2022

## Outside of training - uncomment the line below and delete the above two lines:
##SBATCH --partition=gpu

## Load the CUDA module to access nvcc compiler and CUDA libraires:
module load cuda/11.0

## compile the c+cuda code using the nvcc compiler. Create the 'helloworld' executable program:
nvcc helloworld.cu -o helloworld

## Run the executable program:
./helloworld
