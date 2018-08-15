import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import random

import scipy.misc
import scipy.ndimage
import scipy.stats as st
from os import listdir
from random import shuffle

def gkern(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/np.amax(kernel_raw)#kernel_raw/kernel_raw.sum()
    return kernel

def occluder(x_center,y_center,r,image, fill_val=127, sig_gauss=.7, x_max=None, y_max=None):
    for x in range(x_center-r,x_center+r):
        y = range(int(np.ceil(-np.sqrt(r**2-(x-x_center)**2)+y_center)),int(np.ceil(np.sqrt(r**2-(x-x_center)**2)+y_center)))
        image[x,y]= fill_val# 127 for gray occluders, 0 for black occluders
        
    margin = 5 # Margin can be adjusted
    
    x_left = x_center-r-margin
    y_left = y_center-r-margin
    x_right = x_center+r+margin
    y_right = y_center+r+margin
    if x_left < 0:
        x_left = 0 
    if y_left < 0:
        y_left = 0
    if x_right > x_max:
        x_right = x_max
    if y_right > y_max:
        y_right = y_max
        
    image[x_left:x_right, y_left:y_right]=scipy.ndimage.gaussian_filter(image[x_left:x_right, y_left:y_right],sig_gauss)
    return image