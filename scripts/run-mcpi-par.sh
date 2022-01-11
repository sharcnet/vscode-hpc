#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --time=5
#SBATCH --mem=4G
#SBATCH --cpus-per-task=16
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./mcpi-par 100000000 $SLURM_CPUS_PER_TASK