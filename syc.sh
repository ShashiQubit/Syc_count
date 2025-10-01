#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=10:00:00
#SBATCH --job-name=syc_gpu
#SBATCH --partition=small
#SBATCH --error=syc.%J.err
#SBATCH --output=syc.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashi.kumar@iitgn.ac.in
cd $SLURM_SUBMIT_DIR
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python3 -u syc_count.py
