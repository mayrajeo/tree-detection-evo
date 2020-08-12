#!/bin/bash
#SBATCH --job-name=pack-files
#SBATCH --account=project_2001325
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=small
#SBATCH --mail-type=END
#SBATCH --mail-user=janne.mayra@ymparisto.fi
#SBATCH --cpus-per-task=1
#SBATCH --output=/users/mayrajan/outputs/pack_files_out_%j.txt
#SBATCH --error=/users/mayrajan/outputs/pack_files_err_%j.txt
#SBATCH --gres=nvme:2000

# get data from allas
cd $LOCAL_SCRATCH

# get vnir
mkdir nn-processed-tiles
cd nn-processed-tiles
a-list ibc-carbon-evo-data/nn-processed-tiles/ > tile_list

for f in $(cat tile_list)
do
    a-get $f
done
rm tile_list
cd ..

# move processed tiles back to allas as compressed folder
a-put nn-processed-tiles -b ibc-carbon-evo-data --override
