import pandas as pd
import numpy as np
import os
import itertools
import openscope_predictive_coding as opc
from openscope_predictive_coding import utilities
import random
import hashlib
import json

src_file_name = '/allen/aibs/technology/nicholasc/openscope/NATURAL_SCENES_LUMINANCE_MATCHED.npy'
tgt_dir = '//allen/aibs/technology/nicholasc/openscope'

src_image_data = np.load(src_file_name)

def generate_sequence_block(base_seq, save_file_name=None):
    
    base_seq_str = '_'.join([str(ii) for ii in base_seq])
    if save_file_name is None: 
        save_file_name = os.path.join(tgt_dir, '%s.npy' % base_seq_str)

    N = len(base_seq)
    h, w = src_image_data.shape[1:]
    data_block = np.zeros((N,h,w), dtype=np.uint8)
    for ii, idx in enumerate(base_seq):
        data_block[ii,:,:] = src_image_data[idx,:,:]

    np.save(save_file_name, data_block)
    return hashlib.md5(data_block).hexdigest(), save_file_name

habituation_pilot_checksum_dict = {}

# Generate habituated sequence block: 
habituated_sequence_block_checksum, habituated_sequence_block_file_name = generate_sequence_block(opc.HABITUATED_SEQUENCE_IMAGES)
assert habituated_sequence_block_checksum == 'd1de3dfc46a972d323c7b52b9fff78a9'
habituation_pilot_checksum_dict[habituated_sequence_block_file_name] = habituated_sequence_block_checksum

# Generate randomized oddballs block:
habituation_oddball_full_sequence = utilities.get_shuffled_repeated_sequence(opc.RANDOMIZED_ODDBALL_UNEXPECTED_IMAGES, 100, seed=0)
randomized_oddballs_block_checksum, randomized_oddballs_file_name = generate_sequence_block(habituation_oddball_full_sequence, save_file_name=os.path.join(tgt_dir, 'habituation_randomized_oddball.npy'))
assert randomized_oddballs_block_checksum == 'c38555394253b83f42e2a257e1830c20'
habituation_pilot_checksum_dict[randomized_oddballs_file_name] = randomized_oddballs_block_checksum

# Also add checksum for natural movie 1 from Brain Observatory:
habituation_pilot_checksum_dict[os.path.join(tgt_dir, 'NATURAL_MOVIE_ONE.npy')] = 'b174ad09736c870c6915baf82cf2c9ad'

# Dump to pilot data directory:
json.dump(habituation_pilot_checksum_dict, open(os.path.join(tgt_dir, 'habituation_pilot_checksum.json'), 'w'))

