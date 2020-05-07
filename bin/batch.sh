#!/bin/bash

#SBATCH --job-name=main_progenitors
#SBATCH --mail-user=imendoza@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1                     # number of nodes to use (usually 1)
#SBATCH --ntasks=5                    # how many process do you need?
#SBATCH --cpus-per-task=1             # how many cpus per process?
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=00:30:00
#SBATCH --account=cavestru1
#SBATCH --partition=standard
#SBATCH --output=/home/imendoza/alcca/nbody-relaxed/logs/slurm-%j.out

./bin/run_write_main_line_progenitors.py
