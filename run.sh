#!/bin/sh

## sherlock deployment file that requests 32 cpu cores for 18 hours to run main.py
#SBATCH --job-name=matrix_compressed_sensing
#SBATCH --partition=normal,donoho,hns,stat
#SBATCH --cpus-per-task=64
#SBATCH --time=36:00:00
#SBATCH --error=cs0001.err
#SBATCH --output=cs0001.out


## Run the python script
python3 ./experiment.py
