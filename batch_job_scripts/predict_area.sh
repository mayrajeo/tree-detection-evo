#!/bin/bash
#SBATCH --job-name=predict_area
#SBATCH --account=project_2001325
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END
#SBATCH --mail-user=janne.mayra@ymparisto.fi
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/mayrajan/outputs/predict_area_out_%j.txt
#SBATCH --error=/users/mayrajan/outputs/predict_area_err_%j.txt

# activate environment
source $SCRATCH/mayrajan/conda_activate.sh

# set parameters
tile_fdr=$SCRATCH/mayrajan/aspen_detection/data/tiles
shp_fdr=$SCRATCH/mayrajan/aspen_detection/data/delineated_tiles/
out_fdr=$SCRATCH/mayrajan/aspen_detection/data/results_v3/
ws=2
learn_path=$SCRATCH/mayrajan/aspen_detection/fastai_models/2m_relu_fixed/export.pkl

python $SCRATCH/mayrajan/aspen_detection/predict_fastaiv2.py $tile_fdr $learn_path $shp_fdr $out_fdr $ws

