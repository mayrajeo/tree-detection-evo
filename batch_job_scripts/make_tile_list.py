import os

tiles = [t[:-4] for t in os.listdir('/scratch/project_2001325/mayrajan/data/labeled_tiles/labeled_tiles')]
tiles = list(set(tiles))
paths = [f'ibc-carbon-evo-data/nn-processed-tiles/{t}.tif' for t in tiles]
with open('tile_list', 'w') as f:
    for p in paths:
        f.write(f'{p}\n')
