import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import random
import os
from collections import defaultdict

import scipy.misc
import scipy.ndimage
import scipy.stats as st
from os import listdir
from random import shuffle

import openscope_predictive_coding as opc

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

def get_occlusion_data_metadata(img_num_list, src_image_data, seed=0, metadata=True):
    

    from openscope_predictive_coding.utilities import apply_warp_natural_scene, linear_transform_image

    random.seed(seed)

    # Occlusion:
    img_lummatched = src_image_data
    occlusion_num_dot_list = opc.OCCLUSION_NUM_DOT_LIST
    dot_radius = opc.OCCLUSION_DOT_SIZE
    num_repeats = 10

    img_list = []
    dot_count = -1
    frac_dict = defaultdict(list)
    for num_dots in occlusion_num_dot_list:
        num_dots = int(num_dots)
        dot_count=dot_count+1
        num_dots = int(num_dots)
        
        img_count = -1
        for img_num in img_num_list:
            img_count =img_count+1

            test_img = np.zeros((918,1174))
            x_max = int(np.shape(test_img)[0])
            y_max = int(np.shape(test_img)[1])
            
            for repeat_ii in range(num_repeats):
                
                occluded_img_black = test_img.astype(np.double)
                occluded_img_gray = test_img.astype(np.double)

                r = int(dot_radius)

                # Set x & y grid so that occluding dots are not overlapping.
                x_grid = range(r,x_max-r)#,2*r)
                y_grid = range(r,y_max-r)#,2*r)


                indices = [(m, n) for m in range(len(x_grid)) for n in range(len(y_grid))]
                random_indices = random.sample(indices, num_dots)

                x_dots = [i[0] for i in random_indices]
                y_dots = [i[1] for i in random_indices]

                for i_dot in range(0,num_dots):
                    x_loc = x_dots[i_dot]
                    y_loc = y_dots[i_dot]
                    occluded_img_black = occluder(x_grid[x_loc], y_grid[y_loc], r, occluded_img_black, -1, x_max=x_max, y_max=y_max)
                    occluded_img_gray = occluder(x_grid[x_loc], y_grid[y_loc], r, occluded_img_gray, x_max=x_max, y_max=y_max)

                curr_fraction =  float(len(np.where(occluded_img_black<0)[1]))/np.prod(occluded_img_black.shape)
                frac_dict[num_dots].append(curr_fraction)

                curr_mask = occluded_img_gray
                curr_mast_warp = 1.-linear_transform_image(apply_warp_natural_scene(curr_mask), (0.,254.))

                print curr_fraction, 'num_dots: %s' % num_dots, 'img_num: %s' % img_num, 'repeat_ii: %s' % repeat_ii

                final_image = (img_lummatched[img_num,:,:] - 127.)*curr_mast_warp[::2,::2] + 127.
                img_list.append({'image': final_image,
                                'repeat': repeat_ii,
                                'fraction_occlusion': opc.OCCLUSION_NUM_DOT_to_FRACTION[num_dots],
                                'image_index':img_num})

    random.shuffle(img_list)
    metadata_df_dict = defaultdict(list)
    final_image_list = []
    for curr_image_dict in img_list:
        final_image_list.append(curr_image_dict['image'])
        for key in curr_image_dict:
            if key != 'image':
                metadata_df_dict[key].append(curr_image_dict[key])

    metadata_df = pd.DataFrame(metadata_df_dict)            
    metadata_df.to_csv(os.path.join(opc.data_path, 'natural_scenes_occlusion_warped.csv'))

    data_array = np.array(final_image_list).astype(np.uint8)


    return data_array, metadata_df