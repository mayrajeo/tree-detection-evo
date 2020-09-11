#!/usr/bin/env/python

"""
    Combine shapefiles with treetops and crowns
"""

import sys
import argparse
import os
import re
import geopandas as gpd
from shapely.geometry import Polygon

def preprocess_contour(ttop_fname, crown_fname, min_area, outdir):
    "Combine treetops and crowns, then filter them according to minimum area"
    # Open treetops and edit
    print(f'Processing files {ttop_fname} and {crown_fname}')
    ttops = gpd.read_file(ttop_fname)
    ttops.rename(columns={'geometry':'ttop', 'Z':'max_height'}, inplace=True)
    ttops.set_index('treeID', drop=True, inplace=True)
    ttops['ttop_x'] = ttops.apply(lambda row: row.ttop.x, axis=1)
    ttops['ttop_y'] = ttops.apply(lambda row: row.ttop.y, axis=1)
    ttops = ttops.drop(['ttop', 'max_height'], axis=1)
    # Open crowns
    crowns = gpd.read_file(crown_fname)
    crowns.set_index('value', drop=True, inplace=True)

    crowns.sort_values(by='value', inplace=True)
    # Join dataframes
    crowns = crowns.join(ttops, how='outer')
    crowns = crowns.dropna()
    
    # Fill holes in polygons.
    crowns['geometry'] = crowns.apply(lambda row: Polygon(row.geometry.exterior), axis=1)
    # Add treeID column back
    crowns.rename(columns={'value':'treeID'}, inplace=True)
    
    # Add information about bounding box shapes
    crowns['bounds_x'] = crowns.apply(lambda row: row.geometry.bounds[2] - row.geometry.bounds[0], axis=1)
    crowns['bounds_y'] = crowns.apply(lambda row: row.geometry.bounds[3] - row.geometry.bounds[1], axis=1)
    # Drop all trees with area less than min_area
    crowns = crowns.drop(crowns[crowns.CA_m2 < min_area].index)

    _, tail = os.path.split(ttop_fname)
    tile_id = re.search(r"ttops\_(.*)\.shp", tail).group(1)
    crowns['tile_id'] = tile_id
    crowns.to_file(filename=f'{outdir}/{tile_id}.shp', driver='ESRI Shapefile')
    return 

#from joblib import Parallel, delayed
import multiprocessing


def main(datadir, outdir, min_area):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    treetops = [f for f in os.listdir(datadir) if f.endswith('.shp') and 'ttops' in f]
    crowns = [f for f in os.listdir(datadir) if f.endswith('.shp') and 'crowns' in f]
    treetops.sort()
    crowns.sort()
    #for t, c in zip(treetops, crowns): 
    #    print(f'Processing files {t} and {c}')
    #    preprocess_contour(datadir + t, datadir + c, min_area, outdir)
    # Parallel woo woo
    #Parallel(n_jobs=4)(delayed(preprocess_contour(datadir + t, datadir + c, min_area, outdir))
    #         for t, c in list(zip(treetops, crowns)))
    inputs = [(datadir + t, datadir + c, min_area, outdir) for t, c in zip(treetops, crowns)]
    with multiprocessing.Pool(10) as pool:
        res = pool.starmap(preprocess_contour, inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess shapefiles for data generation')
    parser.add_argument('datadir', type=str, help='path to directory containing shapefiles')
    parser.add_argument('outdir', type=str, help='path to output directory.')
    parser.add_argument('--min_area', type=float, default=10.0,
                        help='Minimum area for treetop, default 10')
    args = parser.parse_args()

    main(args.datadir, args.outdir, args.min_area)
