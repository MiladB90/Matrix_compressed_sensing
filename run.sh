#!/bin/sh

## sherlock deployment file that requests 100 cpu cores for 18 hours to run main.py
#SBATCH --job-name=matrix_completion
#SBATCH --partition=normal,owners,donoho,hns,stat
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --error=mc0014.err
#SBATCH --output=mc0014.out


## Run the python script
python3 ./experiment.py
