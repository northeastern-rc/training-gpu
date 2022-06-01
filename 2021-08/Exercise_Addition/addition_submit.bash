#!/bin/bash 
#SBATCH --job-name=addition
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1    ##request 1 gpu
## Inside training - use the training reservation:
#SBATCH --partition=reservation
#SBATCH --reservation=Training-GPU
## Outside of training - uncomment this line:
##SBATCH --partition=gpu

## Load the CUDA module to load the nvcc compilers and CUDA libraries:
module load cuda/11.1

############ Serial ############

## compile the c+cuda code using the nvcc compiler. Create the 'addition_serial' executable program:
nvcc addition_serial.cu -o addition_serial 
## Run the executable program:
./addition_serial &> addition_serial.output

############ Parallel - Blocks ############

## compile the c+cuda code using the nvcc compiler. Create the 'addition_parallel_blocks' executable program:
nvcc addition_parallel_blocks.cu -o addition_parallel_blocks
## Run the executable program:
./addition_parallel_blocks &> addition_parallel_blocks.output

############ Parallel - Threads ############

## compile the c+cuda code using the nvcc compiler. Create the 'addition_parallel_threads' executable program:
nvcc addition_parallel_threads.cu -o addition_parallel_threads
## Run the executable program:
./addition_parallel_threads &> addition_parallel_threads.output

############ Parallel - Blocks + Threads combined ############

## compile the c+cuda code using the nvcc compiler. Create the 'addition_parallel_combined' executable program:
nvcc addition_parallel_combined.cu -o addition_parallel_combined
## Run the executable program:
./addition_parallel_combined &> addition_parallel_combined.output


