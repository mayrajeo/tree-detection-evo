#!/bin/bash
#SBATCH --job-name=create-train-data
#SBATCH --account=project_2001325
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=small
#SBATCH --mail-type=END
#SBATCH --mail-user=janne.mayra@ymparisto.fi
#SBATCH --cpus-per-task=10
#SBATCH --output=/users/mayrajan/outputs/create_cubes_out_%j.txt
#SBATCH --error=/users/mayrajan/outputs/create_cubes_err_%j.txt
# creates all required datacubes from preprocessed shapefiles and tiles

# activate environment
source $SCRATCH/mayrajan/conda_activate.sh

# get data from allas
#cd $LOCAL_SCRATCH
#mkdir processed-tiles

#cd processed-tiles
#a-list ibc-carbon-evo-data/nn-processed-tiles/ > tile_list
# load only tiles that contain trees
#python $SCRATCH/mayrajan/batch_job_scripts/make_tile_list.py
#for f in $(cat tile_list)
#do
#    a-get $f
#done
#rm tile_list
#cd ..

# set parameters
shp_fn=$SCRATCH/mayrajan/aspen_detection/data/labeled_tiles_fixed/matched_trees.shp
tile_dir=$SCRATCH/data/tiles/
save_dir=$SCRATCH/mayrajan/aspen_detection/data/tree_cubes_10m_fixed

# create non-delineated cubes
for i in {2..5}
do
    python $SCRATCH/mayrajan/aspen_detection/make_train_data.py $shp_fn $tile_dir $save_dir/$i'm' --window_size $i
done

# create non-delineated cubes
for i in {2..5}
do
    python $SCRATCH/mayrajan/aspen_detection/make_train_data.py $shp_fn $tile_dir $save_dir/$i'm_delin' --window_size $i --delin
done
