"""
1. Reads canopy height model and transforms all negative values to NaN
2. Generates image stacks from vnir, swir and chm
    1. Matches tiles with same ID
    2. Upsamples SWIR to have 0.5m ground resolution and transforms to same geotransform as swir and chm
    3. Removes noisy SWIR channels from data
    4. Stacks images into one larger stack so that
        * Bands 1-186 VNIR, 187-460, 461 CHM

"""

import sys
import argparse
import os
import re
import numpy as np 
import xarray as xr 
import rasterio
from affine import Affine
import multiprocessing
from osgeo import gdal, gdalconst

def xarray_to_rasterio(xa, output_filename:str):
    """Converts the given xarray.DataArray object to a raster output file
    using rasterio.
    Arguments:
     - `xa`: The xarray.DataArray to convert
     - `output_filename`: the filename to store the output GeoTIFF file in
    Notes:
    Converts the given xarray.DataArray to a GeoTIFF output file using rasterio.
    This function only supports 2D or 3D DataArrays, and GeoTIFF output.
    The input DataArray must have attributes (stored as xa.attrs) specifying
    geographic metadata, or the output will have _no_ geographic information.
    If the DataArray uses dask as the storage backend then this function will
    force a load of the raw data.
    """
    # Forcibly compute the data, to ensure that all of the metadata is
    # the same as the actual data (ie. dtypes are the same etc)
    xa = xa.load()
    if len(xa.shape) == 2:
        count = 1
        height = xa.shape[0]
        width = xa.shape[1]
        band_indicies = 1
    else:
        count = xa.shape[0]
        height = xa.shape[1]
        width = xa.shape[2]
        band_indicies = np.arange(count) + 1

    processed_attrs = {}

    try:
        val = xa.attrs['affine']
        processed_attrs['affine'] = rasterio.Affine.from_gdal(*val)
    except KeyError:
        pass

    try:
        val = xa.attrs['crs']
        # Our geotiffs don't have crs...
        #processed_attrs['crs'] = rasterio.crs.CRS.from_string(val)
    except KeyError:
        pass

    try:
        val = xa.attrs['transform']
        processed_attrs['transform'] = Affine(*val)
    except KeyError:
        pass

    with rasterio.open(output_filename, 'w',
                       driver='GTiff',
                       height=height, width=width,
                       dtype=str(xa.dtype), count=count,
                       **processed_attrs) as dst:
        dst.write(xa.values, band_indicies)


def resample_swir_gdal(vnir_fn:str, swir_fn:str, order:int):
    if order == 0:
        # order 0 = nearest neighbour
        upsample = gdalconst.GRA_NearestNeighbour
    elif order == 1:
        # order 1 = bilinear
        upsample = gdalconst.GRA_Bilinear
    swirfile = gdal.Open(swir_fn)
    vnirfile = gdal.Open(vnir_fn)
    swir_proj = swirfile.GetProjection()
    vnir_proj = vnirfile.GetProjection()
    swir_trans = swirfile.GetGeoTransform()
    vnir_trans = vnirfile.GetGeoTransform()
    vnir_band = vnirfile.GetRasterBand(1)
    x_vnir = vnirfile.RasterXSize
    y_vnir = vnirfile.RasterYSize
    tile_id = re.search(r"SWIR_(.*)\.tif", swir_fn).group(1)
    outfile = f'temp/SWIR_{tile_id}'
    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create(outfile, x_vnir, y_vnir, 288, vnir_band.DataType)
    output.SetGeoTransform(vnir_trans)
    output.SetProjection(vnir_proj)
    gdal.ReprojectImage(swirfile, output, swir_proj, vnir_proj, upsample)

    # close and remove unneeded files
    del outfile
    del swirfile
    del vnirfile
    os.remove(swir_fn)
    return

def process_tile(chm_fn:str, vnir_fn:str, swir_fn:str, outdir:str, order:int):
    vnir_tile_id = re.search(r"VNIR_(.*)\.tif", vnir_fn).group(1)
    swir_tile_id = re.search(r"SWIR_(.*)\.tif", swir_fn).group(1)
    
    # First upsample SWIR with gdal:
    resample_swir_gdal(vnir_fn, swir_fn, order)

    # Then preprocess chm and stack tiles
    chm = xr.open_rasterio(chm_fn)
    chm.values[chm.values < 0] = np.nan 
    vnir = xr.open_rasterio(vnir_fn)
    temp_swir_fn = f'temp/SWIR_{swir_tile_id}'
    swir = xr.open_rasterio(temp_swir_fn)
    swir = swir[:-14]
    chm_sub = chm.sel(y=slice(max(vnir.y.values), min(vnir.y.values)), 
                      x=slice(min(vnir.x.values), max(vnir.x.values)))

    full_vals = np.vstack((vnir.values, swir.values, chm_sub.values))
    full_vals = full_vals.astype(np.float32)
    attrs = vnir.attrs
    coords = [('band', list(range(1,vnir.shape[0] + swir.shape[0] + chm_sub.shape[0] + 1))),
              ('y', vnir['y'].coords.variables['y']),
              ('x', vnir['x'].coords.variables['x'])]
    full_tile_stack = xr.DataArray(full_vals, coords=coords, 
                                   dims=['band', 'y', 'x'], attrs=attrs)
    xarray_to_rasterio(full_tile_stack, f'{outdir}/{vnir_tile_id}.tif')

    # Close and remove files not needed
    del full_tile_stack
    del vnir
    del swir
    os.remove(vnir_fn)
    os.remove(temp_swir_fn)
    return

def process_all_tiles(chm_file:str, vnir_dir:str, swir_dir:str, outdir:str, order:int):
    if order not in [0,1]:
        print(f'Error! Only order of 0 (NN) or 1 (Bilinear) are supported')
    if not os.path.exists(outdir): os.makedirs(outdir)
    if not os.path.exists('temp'): os.makedirs('temp')
    vnir_tiles = [vnir_dir + f for f in os.listdir(vnir_dir)]
    swir_tiles = [swir_dir + f for f in os.listdir(swir_dir)]

    # Drop unmatching tiles from vnir
    vnir_tiles = [tile for tile in vnir_tiles if not 'R24C20' in tile]
    vnir_tiles = [tile for tile in vnir_tiles if not 'R24C19' in tile]
    
    # Sort just to be sure
    vnir_tiles.sort()
    swir_tiles.sort()

    if len(vnir_tiles) != len(swir_tiles):
        print(f"Error! There are {len(vnir_tiles)} VNIR tiles and {len(swir_tiles)} SWIR tiles.")
        sys.exit(1)

    inputs = [(chm_file, vnir_tiles[i], swir_tiles[i], outdir, order) for i in range(len(vnir_tiles))]
    print(f'Starting to process {len(inputs)} tiles')
    for i in inputs:
        vnir_tile_id = re.search(r"VNIR_(.*)\.tif", i[1]).group(1)
        swir_tile_id = re.search(r"SWIR_(.*)\.tif", i[2]).group(1)
        if vnir_tile_id != swir_tile_id:
            print("Error! Invalid inputs!")
            sys.exit(1)
    with multiprocessing.Pool(10) as pool:
        pool.starmap(process_tile, inputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stack vnir, swir and chm into single files')
    parser.add_argument('chm', type=str, help='Path to canopy height model tiff')
    parser.add_argument('vnir_dir', type=str, 
                        help='Path to the folder containing vnir tiles')
    parser.add_argument('swir_dir', type=str, 
                        help='Path to the folder containing swir tiles')
    parser.add_argument('outdir', type=str,
                        help='Path to output directory')
    parser.add_argument('--order', type=int, default=0,
                        help="""Order of interpolation. Gdal supports only 0 or 1
                                0=Nearest_neighbour,
                                1=Bi-linear""")
    args = parser.parse_args()

    process_all_tiles(args.chm, args.vnir_dir, args.swir_dir, args.outdir, args.order)
