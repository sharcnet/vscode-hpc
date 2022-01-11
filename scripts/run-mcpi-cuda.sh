#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --time=5
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
./mcpi-cuda 100000000