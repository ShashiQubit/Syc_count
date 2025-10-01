#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=10:00:00
#SBATCH --job-name=asyc
#SBATCH --partition=small
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashi.kumar@iitgn.ac.in
cd $SLURM_SUBMIT_DIR
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python3 -u asyc_count.py
