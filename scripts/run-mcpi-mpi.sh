#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --time=5
#SBATCH --ntasks=64         # number of MPI processes
#SBATCH --mem-per-cpu=1024M # memory; default unit is megabytes
srun ./mcpi-mpi 100000000