#!/bin/bash
#SBATCH -J gp_process
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -p gpu-a100-dev
#SBATCH -t 02:00:00

module load gcc/15.2.0
source activate torch_tune

python hyperparam.py