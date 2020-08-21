#!/usr/bn/env/python

"""
Utility functions used in multiple places in various stages of work process
"""

import numpy as np 
from numpy.lib.stride_tricks import as_strided
from math import modf
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import geopandas as gpd

def check_distance(row, df_measured, radius):
    """
    Check if a detected tree has any measured tree within search radius
    """
    x, y = row.X, row.Y
    for testrow in df_measured.itertuples():
        x_test, y_test = testrow.puu_x, testrow.puu_y
        dist = np.sqrt((x-x_test)**2 + (y-y_test)**2)
        if dist <= radius:
            return True
    return False

def get_closest_match(row, df_measured):
    """
    Label detected trees with the closest measured tree from field plots

    NOTE: Should maybe do vice versa, e.g. correct measured trees to nearest detection?
    """
    x, y = row.X, row.Y
    min_rad = 999
    label = 'Undefined'
    for testrow in df_measured.itertuples():
        x_test, y_test = testrow.puu_x, testrow.puu_y
        dist = np.sqrt((x-x_test)**2 + (y-y_test)**2)
        if dist <= min_rad:
            label = testrow.puulaji
            min_rad = dist
    return label

def find_new_coords(row, df_detected):
    x, y = row.puu_x, row.puu_y
    min_rad = 999
    corr_x = 0
    corr_y = 0
    for treetop in df_detected.itertuples():
        x_tree, y_tree = treetop.X, treetop.Y 
        dist = np.sqrt((x-x_tree)**2 + (y-y_tree)**2)
        if dist <= min_rad:
            corr_x = x_tree
            corr_y = y_tree
            min_rad = dist
    return corr_x, corr_y

def label_contours(row, field_plot):
    """Returns original coordinates and species if any are detected within delineated crowns, 
    else returns None"""
    points = [(None, None, None, None, None, None, None)]
    for tree in field_plot.itertuples():
        if Point(tree.tree_X, tree.tree_Y).within(row.geometry.buffer(1)):
            points.append((tree.tree_X, tree.tree_Y, tree.species, 
                           tree.DBH, tree.sum_2019, tree.nov_2019, tree.is_gps))
    if len(points) < 3:
        # No measured trees or only one
        return points[-1]
    else:
        # If multiple points, then return one closest to detected treetop 
        # First element is the dummy value so remove it
        points = points[1:]
        gps_measurements = False
        # Prioritize gps measurements
        for p in points:
            if p[-1] == 1:
                gps_measurements = True
                break
        if gps_measurements:
            gpspoints = [p for p in points if p[-1] == 1]
            points = gpspoints
        # Calculate minimum distance
        min_dist = 9999
        min_idx = 0
        for i, p in enumerate(points):
            dist = np.sqrt((p[0]-row.ttop_x)**2 + (p[1]-row.ttop_y)**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
    return points[min_idx]
    
def round_to_tile(coord):
    """
    Round coordinates to match those in vnir and swir tiles
    """
    a,b = modf(coord)
    if 0 <= a < 0.5:
        return b + 0.25
    return b + 0.75

def filter_too_close(df, radius):
    """
    Filter trees that are too close to each other
    If label is same, drop one of them
    If label is different, drop both
    """
    for row in df.itertuples():
        if row.Index not in df.index:
            continue
        close = df[np.sqrt((df.x - row.x)**2 + (df.y - row.y)**2) < radius]
        close = close.iloc[1:]
        for tree in close.itertuples():
            if row.species == tree.species:
                # Same species, drop later
                df.drop(tree.Index, inplace=True)
            else:
                # Different species, drop both
                # Crashes for some reason, fix?
                df.drop([tree.Index, row.Index], inplace=True)
                #df.drop(row.Index, inplace=True)
                break
    return df

def tile_array(a, b0, b1):
    """
    REPLACED with scipy.ndimage.zoom
    Function to upsample SWIR tiles from 500x500 to 1000x1000 resolution
    Nearest neighbor upsampling. Bilinear or trilinear could be better?
    """
    r, c = a.shape
    rs, cs = a.strides
    x = as_strided(a, (r, b0, c, b1), (rs,0, cs,0))
    return x.reshape(r*b0, c*b1)

def resample_swir(swir_cube):
    """
    REPLACED with scipy.ndimage.zoom
    Upsample SWIR to 0.5m resolution Neares neighbor upsampling
    TODO output and input sizes are expected to be 1000x1000
    and 500x500px, fix to work with pretty much any
    """
    num_chans = swir_cube.shape[0]
    out_arr = np.zeros((num_chans, 1000, 1000))
    for c in range(num_chans):
        out_arr[c] = tile_array(swir_cube[c].values, 2, 2)
    return out_arr

def snv(vals):
    """
    Perform SNV transformation to input tile
    NOTE: Remove all interpolated bands before computing
    """
    vals = (vals - vals.mean(axis=0))/(vals.std(axis=0))
    return vals


def scale_pixels(vals):
    """Scale each pixel with respect to the sum of all bands, 
    as in Dalponte et al 2016
    NOTE: Remove all interpolated bands before computing this
    """
    vals /= vals.sum(axis=0)
    return vals
