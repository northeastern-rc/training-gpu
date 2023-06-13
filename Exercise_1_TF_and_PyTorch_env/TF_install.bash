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

#SBATCH --job-name=TF_build
#SBATCH -N 1
#SBATCH -c 2

# Loading the modules required
module load anaconda3/2022.05 cuda/11.8

conda create --prefix=/scratch/$USER/TF_env_bootcamp python=3.9 -y
source activate /scratch/$USER/TF_env_bootcamp
conda install jupyterlab -y
conda install -c conda-forge cudatoolkit=11.8.0 -y
pip install --upgrade pip
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
