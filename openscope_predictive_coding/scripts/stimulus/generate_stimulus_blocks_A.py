import pandas as pd
import numpy as np
import os
import itertools
import openscope_predictive_coding as opc
from openscope_predictive_coding import utilities
from openscope_predictive_coding.occlusion import occluder

import random
import hashlib
import json

tgt_dir = '//allen/aibs/technology/nicholasc/openscope'
session_type = 'A'

assert os.path.basename(__file__).split('.')[0][-1] == session_type

SEQUENCE_IMAGES = opc.SEQUENCE_IMAGES[session_type]
ODDBALL_IMAGES = opc.ODDBALL_IMAGES[session_type]

oddball_checksum_list = ['4c8ce2cbf60878ac431bb253fe98445a',
                         'd2a3f15f86a5e46c7e4d47f8810ebd9f',
                         'ed10c7974577a9feed0b9dbe99252b8b',
                         '1fefebcbb8921b595b143733cc70a62b',
                         '9c6cdd3fae4ae4b89837c98ebab9b1a0',
                         '386bb661f6f9341478dc08faaef27552',
                         'f9bd13a085941c9763a2997faef1aa14',
                         'ecb8cc06a751e296d978216e41efaaed',
                         '55ad988dabf445690f0e8b60f422cf99',
                         '3e7323094625cc7c552d3d756185dbcd']

pair_checksum_list = ['6a7769be1ba6bc543790fe7adaba5184',
                      '732b6f9d0184ee5f6e9c2d60b363432f',
                      'd6ee977ef0a7c4ae9e060c025de6a206',
                      'd53da0d47537cecceac81828b7e39caf',
                      'fab99098eba1cf6545fcc6b0dca6be76',
                      '667f1b9f76a24bea8008185f08153084',
                      '6ddd447e32ca28def6684e8737319eb0',
                      '59e975e75db9a3e4619b5de754d7dc3c',
                      '91aba83b2f2de0e7b74ef01505f87ee8',
                      'f78142f3f694c48df3cef2adbb940da2',
                      '3cddb740acdaea738a96c75c427d2ae6',
                      'ae0d7f648602809e30735626d9827ea2',
                      '21e6907f3a68b081577de9490fdabab8',
                      '85ef09a9b0bc0a7ab55c592633773403',
                      'e524bec447a47658f9d0282c0d15c35f',
                      '2d567fa740b70e8d6867713444e0da62',
                      '415fda6353725180f63bf2337e144e74',
                      '4e16212f3fd57aab6630aa431e3fa215',
                      'dd0d9b5dd298bd0648b5f457ea64fac6',
                      '56d5d0c5a88ce1fdeea05c2f3fefacfa',
                      'd9db32944a1b6295f63a21ddaf9a3e9c',
                      'e9e92f63db0b160a02df9eb6168479b1',
                      '4186641e1e83ffc8d925e8f1e8e26c60',
                      '07c5c7fbef2df4405227b8ee7768eb2d']

stimulus_pilot_checksum_dict = {}
stimulus_pilot_data = {}

# Generate base sequence:
save_file_name = os.path.join(tgt_dir, '%s.npy' % '_'.join([str(x) for x in SEQUENCE_IMAGES]))
base_block_checksum, base_block_file_name = utilities.generate_sequence_block(SEQUENCE_IMAGES, save_file_name=save_file_name)
assert base_block_checksum == 'd1de3dfc46a972d323c7b52b9fff78a9'
stimulus_pilot_checksum_dict[base_block_file_name] = base_block_checksum

# Generate oddball sequences: 
for curr_checksum, oddball_id in zip(oddball_checksum_list, ODDBALL_IMAGES):
    tmp_seq = [x for x in SEQUENCE_IMAGES]
    tmp_seq[-1] = oddball_id

    save_file_name = os.path.join(tgt_dir, '%s.npy' % '_'.join([str(x) for x in tmp_seq]))
    curr_oddball_block_checksum, curr_oddball_block_file_name = utilities.generate_sequence_block(tmp_seq, save_file_name=save_file_name)
    assert curr_oddball_block_checksum == curr_checksum
    stimulus_pilot_checksum_dict[curr_oddball_block_file_name] = curr_oddball_block_checksum

# Generate transition pair sequences:
pair_list = []
for ii in range(len(SEQUENCE_IMAGES)):
    jj = ii+1
    if jj >= len(SEQUENCE_IMAGES):
        jj = 0
    pair_list.append((ii, jj))
for oddball_image in ODDBALL_IMAGES:
    pair_list.append((SEQUENCE_IMAGES[-2], oddball_image))
    pair_list.append((oddball_image, SEQUENCE_IMAGES[0]))
for pair, curr_checksum in zip(pair_list, pair_checksum_list):
    save_file_name = os.path.join(tgt_dir, '%s.npy' % '_'.join([str(x) for x in pair]))
    pair_checksum, pair_file_name = utilities.generate_sequence_block(pair, save_file_name=save_file_name)
    assert pair_checksum == curr_checksum
    stimulus_pilot_checksum_dict[pair_file_name] = pair_checksum

pair_timing_list = []
pair_timing_dict = utilities.generate_pair_block_timing_dict(pair_list, num_repeats=30, frame_length=.25, expected_duration=360., seed=0)
for key_val in pair_timing_dict.items():
    pair_timing_list.append(key_val)
stimulus_pilot_data['pair_timing'] = pair_timing_list

# Generate randomized oddballs block:
hab_randomized_control_full_sequence = utilities.get_shuffled_repeated_sequence(ODDBALL_IMAGES + SEQUENCE_IMAGES, 30, seed=1)
hab_randomized_control_checksum, hab_randomized_control_file_name = utilities.generate_sequence_block(hab_randomized_control_full_sequence, save_file_name=os.path.join(tgt_dir, 'randomized_control_%s.npy' % session_type))
assert hab_randomized_control_checksum == 'e0639fb53419561ef4f20fa76b016260'
stimulus_pilot_checksum_dict[hab_randomized_control_file_name] = hab_randomized_control_checksum

# Also add checksum for natural movie 1 from Brain Observatory:
stimulus_pilot_checksum_dict[os.path.join(tgt_dir, 'NATURAL_MOVIE_ONE.npy')] = 'b174ad09736c870c6915baf82cf2c9ad'

# Dump to pilot data directory:
json.dump(stimulus_pilot_checksum_dict, open(os.path.join(tgt_dir, 'stimulus_pilot_checksum_dict_A.json'), 'w'))

oddball_list = []
oddball_dict = utilities.generate_oddball_block_timing_dict(SEQUENCE_IMAGES, ODDBALL_IMAGES, expected_duration=2000.0, seed=0)
for key_val in oddball_dict.items():
    oddball_list.append(key_val)
stimulus_pilot_data['oddball_timing'] = oddball_list
json.dump(stimulus_pilot_data, open(os.path.join(tgt_dir, 'stimulus_pilot_data_%s.json' % session_type), 'w'), indent=2)









# img_index = 11

# test_img = utilities.src_image_data[img_index,:,:]
# x_max = int(np.shape(test_img)[0])
# y_max = int(np.shape(test_img)[1])

# num_dots = 30 # Number of occluding dots
# r = 20 # Radius of occluding dots

# # Set x & y grid so that occluding dots are not overlapping.
# x_grid = range(r,x_max-r,2*r)
# y_grid = range(r,y_max-r,2*r)


# indices = [(m, n) for m in range(len(x_grid)) for n in range(len(y_grid))]
# random_indices = random.sample(indices, num_dots)

# x_dots = [i[0] for i in random_indices]
# y_dots = [i[1] for i in random_indices]

# occluded_img = test_img

# for i_dot in range(0,num_dots):
#     x_loc = x_dots[i_dot]
#     y_loc = y_dots[i_dot]
#     occluded_img = occluder(x_grid[x_loc], y_grid[y_loc], r, occluded_img, x_max, y_max)

# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(occluded_img, cmap='gray')
# plt.show()
