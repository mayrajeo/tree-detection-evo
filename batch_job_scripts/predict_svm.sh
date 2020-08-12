#!/bin/bash
#SBATCH --job-name=detect-treetops
#SBATCH --account=project_2001325
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=small
#SBATCH --mail-type=END
#SBATCH --mail-user=janne.mayra@ymparisto.fi
#SBATCH --cpus-per-task=20
#SBATCH --output=/users/mayrajan/outputs/svm_preds_out_%j.txt
#SBATCH --error=/users/mayrajan/outputs/svm_preds_err_%j.txt
#SBATCH --gres=nvme:200

# Generate data
module purge
source $SCRATCH/mayrajan/conda_activate.sh

raw_data_path=$SCRATCH/mayrajan/aspen_detection/data/delineated_tiles_arto
temp_data_path=$LOCAL_SCRATCH/data
temp_pred_path=$LOCAL_SCRATCH/results
outdir=$SCRATCH/mayrajan/aspen_detection/data/results_svm
tile_path=$SCRATCH/data/tiles

cd $LOCAL_SCRATCH
mkdir data
mkdir results

python $SCRATCH/mayrajan/aspen_detection/prepare_svm_data.py $tile_path $raw_data_path $temp_data_path

# load modules

module purge
module load r-env

model_path=$SCRATCH/viinikka/models/svm_vnir_swir_VI_updated.rds

# Run rscript with parameters
Rscript $SCRATCH/mayrajan/aspen_detection/svmPredictions.R $temp_data_path $model_path $temp_pred_path --no-save


# Combine and filter

module purge
source $SCRATCH/mayrajan/conda_activate.sh

borders=$SCRATCH/mayrajan/aspen_detection/data/area_borders.shp

python $SCRATCH/mayrajan/aspen_detection/combine_and_filter_preds.py $temp_pred_path $borders $outdir
