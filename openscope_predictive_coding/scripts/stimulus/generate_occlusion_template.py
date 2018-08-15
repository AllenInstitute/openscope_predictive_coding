import os
import numpy as np
import pandas as pd
from collections import defaultdict

import openscope_predictive_coding as opc
from openscope_predictive_coding.utilities import apply_warp_natural_scene, linear_transform_image
from openscope_predictive_coding.occlusion import occluder
import random

random.seed(0)

# Occlusion:
img_lummatched = opc.get_dataset_template('NATURAL_SCENES_WARPED')
img_num_list = opc.ODDBALL_IMAGES['A']
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

            final_image = (img_lummatched[img_num,::-1,:] - 127.)*curr_mast_warp[::2,::2] + 127.
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
            
pd.DataFrame(metadata_df_dict).to_csv(os.path.join(opc.data_path, 'templates', 'NATURAL_SCENES_OCCLUSION_WARPED.csv'))
np.save(os.path.join(opc.data_path, 'templates', 'NATURAL_SCENES_OCCLUSION_WARPED.npy'),  np.array(final_image_list, dtype=np.uint8)) 



# for key, val in sorted(frac_dict.items(), key=lambda x: x[0]):
#     print key, np.mean(val)