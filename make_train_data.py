#!/usr/bin/env/python

"""
Extracts spectral cubes from raw image tiles and chm.

Usage info:

python make_train_data.py -h

Does the following:

1. Reads labeled shapefiles and matches them to stacked tiles



"""

import sys
import argparse
import os
import re
import xarray as xr
import numpy as np 
import pandas as pd 
import geopandas as gpd
from math import modf
from src import utils
from scipy.ndimage import zoom
from shapely.geometry import Polygon, Point
from itertools import product
import multiprocessing

def generate_cubes_from_tile(tile_fn, trees_in_tile, save_dir, ws, delineate=False, normalize=False):
    tile = xr.open_rasterio(tile_fn)
    print(f'Processing tile {tile_fn}, {len(trees_in_tile)} trees to extract')
    for tree in trees_in_tile.itertuples():
        cropped = tile.sel(y=slice(tree.ttop_y + ws, tree.ttop_y - ws ),
                           x=slice(tree.ttop_x - ws,  tree.ttop_x + ws)).copy()
        if cropped.shape[1] != ws*4 + 1: continue

        if cropped.shape[2] != ws*4 + 1: continue
        if normalize:
            cropped[:-1] = cropped[:-1] / cropped[:-1].sum(axis=0)
        if delineate:
            for x,y in product(range(cropped.shape[2]), range(cropped.shape[1])):
                if not Point(cropped[:,y,x].x.values, cropped[:,y,x].y.values).within(tree.geometry):
                    cropped[:,y,x] = np.nan
        np.save(f'{save_dir}/{tree.filename}', cropped.values)
    return


def main_func(shp_fn, tile_dir, save_dir, window_size, delineate, normalize):
    """
    Main function for training data generation
    """
    # We have a preprocessed dataframe containing the trees
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    all_trees = gpd.read_file(shp_fn)
    inputs = [(f'{tile_dir}/{t}.tif', all_trees[all_trees.tile_id == t], 
              save_dir, window_size, delineate, normalize) for t in all_trees.tile_id.unique()]
    with multiprocessing.Pool(10) as pool:
        pool.starmap(generate_cubes_from_tile, inputs)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set parameters for data generation')
    parser.add_argument('shp_fn', type=str, 
                        help='Filename containing labeled tree locations')
    parser.add_argument('tile_dir', type=str, 
                        help='Path to the folder containing stacked tiles')
    parser.add_argument('save_dir', type=str, 
                        help='Path to the output directory.')
    # Optional
    parser.add_argument('--window_size', type=float, default=4.0,
                        help='Radius of the square extracted around treetops. Default 4m')

    # Mask areas outside delineated tree?
    delin_parser = parser.add_mutually_exclusive_group(required=False)
    delin_parser.add_argument('--delin', dest='delineate', action='store_true',
                        help='Mask points outside of tree crown.')
    delin_parser.add_argument('--no-delin', dest='delineate', action='store_false',
                        help='No delineation masking, Default')
    norm_parser = parser.add_mutually_exclusive_group(required=False)
    norm_parser.add_argument('--norm', dest='normalize', action='store_true',
                        help='Normalize points in respect to total sum.')
    norm_parser.add_argument('--no-norm', dest='normalize', action='store_false',
                        help='No normalization, Default')
    parser.set_defaults(normalize=False, delineate=False)
    args = parser.parse_args()

    main_func(args.shp_fn, args.tile_dir, args.save_dir, 
              args.window_size, args.delineate, args.normalize)
