#!/bin/sh

## sherlock deployment file that requests 32 cpu cores for 18 hours to run main.py
#SBATCH --job-name=cs01
#SBATCH --partition=normal,donoho,hns,stat,bigmem
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G  # memory per cpu-core
#SBATCH --time=24:00:00
#SBATCH --error=cs0001.err
#SBATCH --output=cs0001.out


## Run the python script
python3 ./experiment.py

