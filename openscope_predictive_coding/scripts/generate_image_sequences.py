import pandas as pd
import numpy as np
import os
import itertools
import openscope_predictive_coding as opc
from openscope_predictive_coding import utilities
import random

src_file_name = '/allen/aibs/technology/nicholasc/openscope/NATURAL_SCENES_LUMINANCE_MATCHED.npy'
tgt_dir = '/allen/aibs/technology/nicholasc/openscope'

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

# Generate habituated sequence block: 
generate_sequence_block(opc.HABITUATED_SEQUENCE_IMAGES)

# Generate randomized oddballs block:
habituation_oddball_full_sequence = utilities.get_shuffled_repeated_sequence(opc.RANDOMIZED_ODDBALL_UNEXPECTED_IMAGES, 100, seed=0)
generate_sequence_block(habituation_oddball_full_sequence, save_file_name=os.path.join(tgt_dir, 'habituation_randomized_oddball.npy'))

