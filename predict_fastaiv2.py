from fastai2.basics import *
from fastai2.vision.augment import *
from fastai2.vision.core import *
from fastai2.vision.data import *
from fastai2.data.all import *
from src.multichannel import *
from src.customnets import SimpleNet
import geopandas as gpd
import xarray as xr
import argparse
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

def listrange(start, end): return list(range(start,end))

SELECTION = listrange(0,100) + listrange(101, 155) + listrange(195, 209) + listrange(230,255) + listrange(280,337)


def get_trees(df, tile, window_size):
    cubes = []
    for tree in df.itertuples():
        ymax_tile = max(tile.coords.variables['y'].values)
        ymin_tile = min(tile.coords.variables['y'].values)
        xmax_tile = max(tile.coords.variables['x'].values)
        xmin_tile = min(tile.coords.variables['x'].values)
        x_coord = tree.ttop_x
        y_coord = tree.ttop_y
        ymax = y_coord + window_size
        ymin = y_coord - window_size
        xmax = x_coord + window_size
        xmin = x_coord - window_size
        cropped = tile.sel(y=slice(ymax, ymin), x=slice(xmin, xmax)).copy()
        cropped = cropped.values

        # Check that square image was extracted
        if cropped.shape[1] != (4*window_size)+1 or cropped.shape[2] != (4*window_size)+1: 
            # Check where to pad
            x_left, x_right, y_top, y_bot = 0,0,0,0
            if ymax > ymax_tile:
                y_top = int((ymax - ymax_tile) * 2)
            if xmax > xmax_tile:
                x_right = int((xmax - xmax_tile) * 2)
            if ymin < ymin_tile:
                y_bot = int((ymin_tile - ymin) * 2)
            if xmin < xmin_tile:
                x_left = int((xmin_tile - xmin) * 2)
            cropped = np.pad(cropped, ((0,0), (y_top, y_bot), (x_left, x_right)), 'reflect')
        cubes.append(MultiChannelTensorImage.create(cropped))
    return cubes

def predict_tile(learn:Learner, tile_fn:str, shp_fn:str, outdir:str, ws:int=3):
    
    shape = gpd.read_file(shp_fn)
    tile = xr.open_rasterio(tile_fn)[SELECTION]
    tile_dl = learn.dls.test_dl(get_trees(shape, tile, ws), with_labels=False)
    preds = learn.get_preds(ds_idx=0, dl=tile_dl, with_decoded=True)
    shape['prediction'] = [learn.dls.vocab[i] for i in preds[2]]
    tile_id = shape.iloc[0].tile_id
    shape.to_file(f'{outdir}/{tile_id}.shp', driver='ESRI Shapefile')
    pine_count = len(shape[shape.prediction == 'Scots pine'])
    spruce_count = len(shape[shape.prediction == 'Norway spruce'])
    birch_count = len(shape[shape.prediction == 'Birch'])
    aspen_count = len(shape[shape.prediction == 'European aspen'])
    return pine_count, spruce_count, birch_count, aspen_count

def predict_batch(tile_fdr:str, learn_fdr:str, shape_fdr:str, outdir:str, ws:int=3):
    tiles = [f for f in os.listdir(tile_fdr) if f.endswith('.tif')]
    shapes = [s for s in os.listdir(shape_fdr) if s.endswith('.shp')]
    learn = load_learner(learn_fdr, cpu=False)
    tiles.sort()
    shapes.sort()
    assert(len(tiles) == len(shapes))
    # Init counts
    pine_count = 0
    spruce_count = 0
    birch_count = 0
    aspen_count = 0
    tot_count = 0
    tile_count = 0
    for i in range(len(tiles)):
        print(f'processing {tiles[i]}')
        pc, sc, bc, ac = predict_tile(learn, f'{tile_fdr}/{tiles[i]}', f'{shape_fdr}/{shapes[i]}', outdir, ws)
        pine_count += pc
        spruce_count += sc
        birch_count += bc
        aspen_count += ac
        tot_count += (pc + sc + bc + ac)
    
    # Print report of tree proportions
    print(f"""Summary of predictions:
              Scots pine: {(pine_count / tot_count):.4f}
              Norway spruce: {(spruce_count / tot_count):.4f}
              Birch: {(birch_count / tot_count):.4f}
              European aspen: {(aspen_count / tot_count):.4f}
    """)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set parameters for batch inference')
    parser.add_argument('tile_fdr', type=str, help='Path to preprocessed tiles')
    parser.add_argument('learn_fdr', type=str, help='Path for trained learner')
    parser.add_argument('shape_fdr', type=str, help='Path to preprocessed shapefiles')
    parser.add_argument('outdir', type=str, help='Path to output directory')
    parser.add_argument('ws', type=int, help='Used window size')
    parser.set_defaults(delineate=False, normalize=False)
    args = parser.parse_args()
    predict_batch(args.tile_fdr, args.learn_fdr, args.shape_fdr, args.outdir, args.ws) 
