#!/bin/bash
#SBATCH --job-name=detect-treetops
#SBATCH --account=project_2001325
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=small
#SBATCH --mail-type=END
#SBATCH --mail-user=janne.mayra@ymparisto.fi
#SBATCH --cpus-per-task=10
#SBATCH --output=/users/mayrajan/outputs/detect_trees_out_%j.txt
#SBATCH --error=/users/mayrajan/outputs/detect_trees_err_%j.txt

# load modules
module load r-env
pwd
# Move to local scratch space
#cd $SCRATCH
#pwd
# Get data from allas 
#mkdir processed-tiles
#cd processed-tiles
#a-list ibc-carbon-evo-data/nn-processed-tiles/ > tile_list
#for f in $(cat tile_list)
#do
#    a-get $f
#done
#rm tile_list
#cd ..
# set parameters
data_path=$SCRATCH/data/tiles
ws=5
hmin=15
outdir=$SCRATCH/mayrajan/aspen_detection/data/delineated_tiles_arto/raw/

# Run rscript with parameters
Rscript $SCRATCH/mayrajan/aspen_detection/getContoursDalponte.R $data_path $ws $hmin $outdir --no-save
