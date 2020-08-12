#!/bin/bash
#SBATCH --job-name=create-train-data-hist
#SBATCH --account=project_2001325
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=small
#SBATCH --mail-type=END
#SBATCH --mail-user=janne.mayra@ymparisto.fi
#SBATCH --cpus-per-task=10
#SBATCH --output=/users/mayrajan/outputs/create_cubes_hist_out_%j.txt
#SBATCH --error=/users/mayrajan/outputs/create_cubes_hist_err_%j.txt
#SBATCH --gres=nvme:500
# creates all required datacubes from preprocessed shapefiles and tiles 
# and makes histogram matching 

# activate environment
source $SCRATCH/mayrajan/conda_activate.sh

# set parameters
shp_fn=$SCRATCH/mayrajan/aspen_detection/data/labeled_tiles_arto/matched_trees.shp
tile_dir=$SCRATCH/data/tiles/
hist_tile_dir=$LOCAL_SCRATCH/tiles/
save_dir=$SCRATCH/mayrajan/aspen_detection/data/tree_cubes_hist/
ref_tileid=R13C12

echo 'starting histogram matching'

# Do histogram matching
python $SCRATCH/mayrajan/aspen_detection/histogram_matching.py $tile_dir $hist_tile_dir $ref_tileid $shp_fn

echo 'Histogram matching ready'

# create non-delineated cubes
for i in {1..5}
do
    python $SCRATCH/mayrajan/aspen_detection/make_train_data.py $shp_fn $hist_tile_dir $save_dir/$i'm' --window_size $i
done
