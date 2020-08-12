#!/bin/bash
#SBATCH --job-name=match-treetops
#SBATCH --account=project_2001325
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=small
#SBATCH --mail-type=END
#SBATCH --mail-user=janne.mayra@ymparisto.fi
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/mayrajan/outputs/match_treetops_out_%j.txt
#SBATCH --error=/users/mayrajan/outputs/match_treetops_err_%j.txt

# activate environment
source $SCRATCH/mayrajan/conda_activate.sh
scratchdir=$SCRATCH/mayrajan/aspen_detection

cd $scratchdir/
# set parameters for preprocessing
data_path=$scratchdir/data/delineated_tiles_arto/raw/
outdir=$scratchdir/data/delineated_tiles_arto/
min_area=1
# Run match treetops and delineations
#python preprocess_shapefiles.py $data_path $outdir --min_area $min_area

# set parameters for tree matching
field_data=$scratchdir/data/field_data/all_trees_dbh_150_dec_2019.shp
tree_crown_dir=$scratchdir/data/delineated_tiles_arto/
shp_out_dir=$scratchdir/data/labeled_tiles_arto/
# Run tree detection
python match_field_data.py $field_data $tree_crown_dir $shp_out_dir
