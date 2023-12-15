#!/bin/sh

## farmshare deployment file that requests 16 cpu cores for 15 minutes to run main.py
#SBATCH --job-name=matrix_completion
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --error=mc0013.err
#SBATCH --output=mc0013.out


## Run the python script
time python3 ./experiment.py
