#!/bin/bash 
#SBATCH --job-name=Hello
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1	##request 1 gpu
#SBATCH --partition=reservation
#SBATCH --reservation=bootcamp2021v100

# Use gpu partition when not in training. Uncomment the line below and delete the above two lines.                                                                 
# #SBATCH --partition=gpu

module load cuda/11.0
module load anaconda3/2021.05

nvcc helloworld.cu -o helloworld
./helloworld
