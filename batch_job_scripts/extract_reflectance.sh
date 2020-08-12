#!/bin/bash
#SBATCH --job-name=extract_reflectace
#SBATCH --account=project_2001325
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=small
#SBATCH --mail-type=END
#SBATCH --mail-user=janne.mayra@ymparisto.fi
#SBATCH --cpus-per-task=10
#SBATCH --error=/users/mayrajan/outputs/extract_reflectance_err_%j.txt
#SBATCH --output=/users/mayrajan/outputs/extract_reflectance_out_%j.txt

# load modules
module load r-env

# Run rscript with parameters
Rscript /scratch/project_2001325/mayrajan/extract_hs_data.R --no-save
