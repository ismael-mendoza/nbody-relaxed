#!/bin/bash

#SBATCH --job-name=main_progenitors
#SBATCH --mail-user=imendoza@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=500MB
#SBATCH --time=00:30:00
#SBATCH --account=cavestru
#SBATCH --partition=standard
#SBATCH --output=/home/imendoza/alcca/%u/%x-%j.log

./bin/run_write_main_line_progenitors.py