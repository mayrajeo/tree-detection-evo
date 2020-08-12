import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple

def scale_channel(chan:np.ndarray, vmin:float=None, vmax:float=None) -> np.ndarray:
    "Scale single channel to interval 0 and 1"
    if vmin is None: vmin = np.percentile(chan, 1)
    if vmax is None: vmin = np.percentile(chan, 99)
    outchan = (chan - vmin) / (vmax - vmin)
    outchan[outchan < 0] = 0.
    outchan[outchan > 1] = 1.
    outchan[~np.isfinite(outchan)] = 0
    return outchan

def scale_image(img:np.ndarray, channels:Tuple[int, int, int]=(82,49,28), 
                vmin:Tuple[float,float,float]=None, vmax:Tuple[float,float,float]=None) -> np.ndarray:
    "Scale input img array to range 0-1. Assumes channels first -format"
    r, g, b = channels
    if vmin is None: 
        vmin = (np.percentile(img[r], 1), np.percentile(img[g], 1), np.percentile(img[b],1))
    if vmax is None: 
        vmax = (np.percentile(img[r], 99), np.percentile(img[g], 99), np.percentile(img[b],99))
    im = np.zeros((img.shape[1], img.shape[2], 3))
    im[...,0] = scale_channel(chan=img[r], vmin=vmin[0], vmax=vmax[0])
    im[...,1] = scale_channel(chan=img[g], vmin=vmin[1], vmax=vmax[1])
    im[...,2] = scale_channel(chan=img[b], vmin=vmin[2], vmax=vmax[2])
    im[im < 0] = 0
    im[im > 1] = 1
    im[~np.isfinite(im)] = 0
    return im

def calc_quantiles(img:np.ndarray, channels:Tuple[int, int, int]=(82,49,28), 
                   min_q:int=1, max_q:int=99) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    "Calculate min and max quantiles for img rescaling"
    r, g, b = channels
    vmin = np.percentile(img[[r,g,b]], min_q, axis=(1,2))
    vmax = np.percentile(img[[r,g,b]], max_q, axis=(1,2))
    return vmin, vmax

def show_image(img:np.ndarray, channels:Tuple[int, int, int]=(82,49,28), 
               vmin:Tuple[float, float, float]=None, 
               vmax:Tuple[float, float, float]=None, 
               ax:plt.Axes=None, figsize:tuple=(3,3),
               hide_axis:bool=False, **kwargs) -> plt.Axes:
    "Show three channel image on ax."
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    im = scale_image(img=img, channels=channels, vmin=vmin, vmax=vmax)
    ax.imshow(im)
    if hide_axis:
        ax.set_xticks([])
        ax.set_yticks([])
    return ax

def calculate_spectral_index(img:np.ndarray, channels:Tuple[int, int]=(82,142)) -> np.ndarray:
    """Calculates normalized spectral index based on two channels. 
    Example: for NDVI, first channel is assumed to be red and second NIR
    """
    r, nir = channels
    spectral_index = (img[nir] - img[r]) / (img[nir] + img[r])
    return spectral_index

def plot_chm_contour(chm:np.ndarray, ax:plt.Axes=None, **kwargs) -> plt.Axes:
    "Plot chm as contour function"
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    xs = range(chm.shape[-2])
    ys = range(chm.shape[-1], 0, -1)
    X, Y = np.meshgrid(xs, ys)
    top = np.amax(chm)
    bot = np.amin(chm)
    cs = ax.contourf(X, Y, chm, levels=np.linspace(bot, top, num=100, endpoint=True))
    return ax 
