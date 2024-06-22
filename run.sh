#!/bin/sh

## sherlock deployment file that requests 32 cpu cores for 18 hours to run main.py
#SBATCH --job-name=cs01
#SBATCH --partition=normal,donoho,hns,stat,owners
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --error=cs0001.err
#SBATCH --output=cs0001.out


## Run the python script
python3 ./experiment.py
