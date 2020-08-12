#!/bin/bash
#SBATCH --job-name=resample-tiles
#SBATCH --account=project_2001325
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=small
#SBATCH --mail-type=END
#SBATCH --mail-user=janne.mayra@ymparisto.fi
#SBATCH --cpus-per-task=10
#SBATCH --output=/users/mayrajan/outputs/stack_tiles_out_%j.txt
#SBATCH --error=/users/mayrajan/outputs/stack_tiles_err_%j.txt
#SBATCH --gres=nvme:2000
# creates all required datacubes from preprocessed shapefiles and tiles

# activate environment
source $SCRATCH/mayrajan/conda_activate.sh

# get data from allas
cd $LOCAL_SCRATCH

# get vnir
mkdir vnir
cd vnir
a-list ibc-carbon-evo-data/vnir/ > tile_list

#cnt=0
for f in $(cat tile_list)
do
    a-get $f
    #((cnt++))
    #if [ $cnt -eq 3 ]; then
    #    break
    #fi
done
rm tile_list
cd ..

# get swir
mkdir swir
cd swir
a-list ibc-carbon-evo-data/swir/ > tile_list
#cnt=0
for f in $(cat tile_list)
do
    a-get $f
    #((cnt++))
    #if [ $cnt -eq 3 ]; then
    #    break
    #fi
done
rm tile_list
cd ..

# get chm
a-get ibc-carbon-evo-data/chm/ChmEvo_norm.tif

# make directory for upsampled tiles
mkdir nn-processed-tiles

# set parameters
chm=$LOCAL_SCRATCH/ChmEvo_norm.tif
vnir=$LOCAL_SCRATCH/vnir/
swir=$LOCAL_SCRATCH/swir/
outdir=$LOCAL_SCRATCH/nn-processed-tiles/
# Order: 0=nearest neighbor, 1=bi-linear, 2=bi-quadratic 3=bi-cubic, 4=bi-quartic, 5=bi-quintic
order=0

# stack tiles
python $SCRATCH/mayrajan/stack_tiles_gdal.py $chm $vnir $swir $outdir --order $order

# these are not needed anymore, so remove them
rm -rf vnir
rm -rf swir
rm -rf ChmEvo_norm.tif

# move processed tiles back to allas one by one. use rclone because for this its easier
source /appl/opt/allas-cli-utils/allas_conf -f -k $OS_PROJECT_NAME
rclone copyto nn-processed-tiles/ allas:ibc-carbon-evo-data/nn-processed-tiles -v

# move processed tiles back to allas as compressed folder
a-put nn-processed-tiles -b ibc-carbon-evo-data
