import pandas as pd
import numpy as np
import os
import itertools
import openscope_predictive_coding as opc
from openscope_predictive_coding.utilities import generate_sequence_block, seq_to_str, get_hash, get_shuffled_repeated_sequence
import random
import hashlib
import json

tgt_dir = curr_path = os.path.join(opc.data_path)
src_image_data = opc.stimulus.get_stimulus_template('natural_scenes_warped')

# Generate habituated sequence block: 
habituated_sequence_data = generate_sequence_block(opc.HABITUATED_SEQUENCE_IMAGES, src_image_data)
save_file_name = os.path.join(tgt_dir, "%s_%s.npy" % (seq_to_str(opc.HABITUATED_SEQUENCE_IMAGES), get_hash(habituated_sequence_data)))
print save_file_name
# np.save(save_file_name, habituated_sequence_data)


# # Generate randomized oddballs block:
# habituation_oddball_full_sequence = get_shuffled_repeated_sequence(opc.HABITUATED_ODDBALL_IMAGES, 100, seed=0)
# randomized_oddballs_data = generate_sequence_block(habituation_oddball_full_sequence, src_image_data)
# save_file_name = os.path.join(tgt_dir, 'habituation_pilot_randomized_oddball_%s.npy' % get_hash(randomized_oddballs_data))
# print save_file_name
# np.save(save_file_name, randomized_oddballs_data)
