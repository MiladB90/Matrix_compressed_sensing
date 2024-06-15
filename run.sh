#!/bin/sh

## sherlock deployment file that requests 32 cpu cores for 18 hours to run main.py
#SBATCH --job-name=matrix_completion
#SBATCH --partition=normal,owners,donoho,hns,stat
#SBATCH --cpus-per-task=64
#SBATCH --time=36:00:00
#SBATCH --error=mc0016.err
#SBATCH --output=mc0016.out


## Run the python script
python3 ./experiment.py
