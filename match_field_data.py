#!/usr/bin/env/python

"""
Match field measurements to detected trees
Usage info:

python match_to_contour.py -h

Does the following:

1. Read preprocessed shapefiles for each tile
2. For each tile, do the following:
    1. Read shapefile containing treetop locations and crown contours
    2. Read all measured trees that are located within the tile
    3. Check which of the measured trees are located within any contours 
       and move their coordinates to corresponding treetops
    4. If two or more trees are inside same tree crown, then
        1. prioritize gps measurements over other
        2. select measurement nearest to detected treetop
    5. Save shapefiles to specified directory

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

def generate_data_contour(field_measurements,
                          tree_crown_dir, output_directory):
    """
    Main function for training data generation
    """
    
    # Read shapefile containing field measurements
    if os.path.splitext(field_measurements)[1] == '.shp':
        trees_shp = gpd.read_file(field_measurements)
    elif os.path.splitext(field_measurements)[1] == '.csv':
        trees_shp = pd.read_csv(field_measurements)
    else:
        print('Currently only .shp and .csv are supported')
        sys.exit(1)
    trees_shp = trees_shp[['species', 'tree_X', 'tree_Y', 'DBH', 'nov_2019', 'sum_2019', 'is_gps']]
    trees_shp.drop_duplicates(['tree_X', 'tree_Y'], inplace=True)
    tiles = [tree_crown_dir + f for f in os.listdir(tree_crown_dir) if f.endswith('.shp')]
    tiles.sort()
    tiles = tiles
    tree_id_nr = 0

    # Create outdir if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(f'{output_directory}/labeled_tiles/'):
        os.makedirs(f'{output_directory}/labeled_tiles/')

    for t in tiles:
        # Extract tile_id
        _, tail = os.path.split(t)
        tile_id = re.search(r"(.*)\.shp", tail).group(1)
        print(f'Processing tile {tile_id}')

        # Read delineated crowns from shp
        tile_lidar_detected = gpd.read_file(f'{tree_crown_dir}/{tile_id}.shp')
        xdims = tile_lidar_detected.ttop_x.min(), tile_lidar_detected.ttop_x.max()
        ydims = tile_lidar_detected.ttop_y.min(), tile_lidar_detected.ttop_y.max()
        # Filter treetops
        tile_field_plot = trees_shp[trees_shp['tree_Y'].between(ydims[0], ydims[1]) & 
                                    trees_shp['tree_X'].between(xdims[0], xdims[1])].copy()
        if len(tile_field_plot) == 0:
            print(f'No measured trees in tile {tile_id}')
            continue

        # Match lidar and field plot trees
        if len(tile_lidar_detected) == 0:
            print(f'No detected trees in tile {tile_id}')
            continue
        
        #tile_field_plot['crownID'] = tile_field_plot.apply(lambda row: utils.find_crown(row, tile_lidar_detected), axis=1)
        tile_lidar_detected[['meas_x', 'meas_y', 'species', 'dbh', 'sum_2019', 'nov_2019', 'is_gps']] = tile_lidar_detected.apply(lambda row: utils.label_contours(row, tile_field_plot), axis=1, result_type='expand')
        # For inspection, save shapefile with species information for each tile
        tile_lidar_detected.to_file(filename=f'{output_directory}labeled_tiles/{tile_id}.shp', driver='ESRI Shapefile')

        # Drop trees that were not labeled
        tile_lidar_detected = tile_lidar_detected.dropna()

        print(f'{len(tile_lidar_detected)} remain after correction and drop')
        # Extract data cubes for each tree. Do not normalize any other channel except CHM one.
        for tree in tile_lidar_detected.itertuples():
            # Save tree information
            if 'tree_shapes' not in locals():
                tree_shapes = gpd.GeoDataFrame([tree])
            else:
                tree_shapes = tree_shapes.append([tree], ignore_index=True)

            tree_id_nr += 1
    tree_shapes.drop('Index', inplace=True, axis=1)
    tree_shapes['filename'] = [f'{i}.npy' for i in range(len(tree_shapes))]
    # Finally save a data frame containing the information of all detected trees
    tree_shapes.to_file(filename=f'{output_directory}/matched_trees.shp', driver='ESRI Shapefile')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set parameters for data generation')
    parser.add_argument('field_measurements', type=str, 
                        help='Path to the file containing the field measurements')
    parser.add_argument('tree_crown_dir', type=str, 
                        help='Directory containing shapefiles for delineated trees')
    parser.add_argument('output_directory', type=str, 
                        help='Path to the output directory.')
    args = parser.parse_args()

    generate_data_contour(args.field_measurements, args.tree_crown_dir, 
                          args.output_directory)
