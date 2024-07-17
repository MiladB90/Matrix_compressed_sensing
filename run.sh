#!/bin/sh

## sherlock deployment file that requests 32 cpu cores for 18 hours to run main.py
#SBATCH --job-name=cs02
#SBATCH --partition=normal,donoho,hns,stat
#SBATCH --cpus-per-task=128

##SBATCH --mem-per-cpu=2G  # memory per cpu-core

#SBATCH --time=48:00:00
#SBATCH --error=cs0002.err
#SBATCH --output=cs0002.out


## Run the python script
python3 ./experiment.py

