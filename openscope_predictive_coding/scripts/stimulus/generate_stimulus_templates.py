import numpy as np
import os

import openscope_predictive_coding as opc
from openscope_predictive_coding.utilities import apply_warp_natural_scene, apply_warp_natural_movie, run_camstim_debug, downsample_monitor_to_template, luminance_match, get_hash
import allensdk.brain_observatory.stimulus_info as si


template_dict = opc.stimulus.get_stimulus_template(si.NATURAL_MOVIE_ONE)

raise

# Build natural movie one:
nm1_data = template_dict[si.NATURAL_MOVIE_ONE]
nm1_warp_path = os.path.join(opc.data_path, 'templates', 'NATURAL_MOVIE_ONE_WARPED_77ee4ecd0dc856c80cf24621303dd080.npy')
if os.path.exists(nm1_warp_path):
    nm1_data_warp = np.load(nm1_warp_path)
    raise
    assert get_hash(nm1_data_warp) == '77ee4ecd0dc856c80cf24621303dd080'
else:
    nm1_data_warp = np.empty((nm1_data.shape[0], opc.IMAGE_H, opc.IMAGE_W), dtype=np.uint8)
    for fi, img in enumerate(nm1_data):
        print fi, 'NM1'
        img_warp = apply_warp_natural_movie(img)
        nm1_data_warp[fi,:,:] = downsample_monitor_to_template(img_warp)[::-1,:]
    np.save(nm1_warp_path, nm1_data_warp)
    

# Build natural movie two:
nm2_data = template_dict[si.NATURAL_MOVIE_TWO]
nm2_warp_path = os.path.join(opc.data_path, 'templates', 'NATURAL_MOVIE_TWO_WARPED_92f1fe36e2c118761cbebcebcc6cd076.npy')
if os.path.exists(nm2_warp_path):
    nm2_data_warp = np.load(nm2_warp_path)
    raise
    assert get_hash(nm2_data_warp) == ''
else:
    nm2_data_warp = np.empty((nm2_data.shape[0], opc.IMAGE_H, opc.IMAGE_W), dtype=np.uint8)
    for fi, img in enumerate(nm2_data):
        print fi, 'NM2'
        img_warp = apply_warp_natural_movie(img)
        nm2_data_warp[fi,:,:] = downsample_monitor_to_template(img_warp)[::-1,:]
    np.save(nm2_warp_path, nm2_data_warp)


# Warp and luminance match natural scenes:
ns_data = template_dict[si.NATURAL_SCENES]
ns_warp_path = os.path.join(opc.data_path, 'templates', 'NATURAL_SCENES_WARPED_8ba4262b06ec81c3ec8d3d7d7831e564.npy')
if os.path.exists(ns_warp_path):
    ns_data_warp = np.load(ns_warp_path)
    raise
    assert get_hash(ns_data_warp) == ''
else:
    ns_data_warp_lm = np.empty((ns_data.shape[0], opc.IMAGE_H, opc.IMAGE_W), dtype=np.uint8)
    for fi, img in enumerate(ns_data):
        img_warp = apply_warp_natural_scene(img)
        img_warp_lum = luminance_match(img_warp)
        print fi, 'NS', img_warp_lum.mean(), img_warp_lum.std()/img_warp_lum.mean()
        ns_data_warp_lm[fi,:,:] = downsample_monitor_to_template(img_warp_lum)[::-1,:]
    np.save(os.path.join(opc.data_path, 'templates', 'NATURAL_SCENES_WARPED.npy'), ns_data_warp_lm)

