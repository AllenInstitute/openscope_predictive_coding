import pandas as pd
import numpy as np
import os
import itertools
import openscope_predictive_coding as opc
from openscope_predictive_coding.utilities import generate_sequence_block, seq_to_str, get_hash, get_shuffled_repeated_sequence
import random
import hashlib
import json

tgt_dir = curr_path = os.path.join(opc.data_path, 'sequences')
src_image_data = opc.get_dataset_template('NATURAL_SCENES_WARPED')

# Generate habituated sequence block: 
habituated_sequence_data = generate_sequence_block(opc.HABITUATED_SEQUENCE_IMAGES, src_image_data)
save_file_name = os.path.join(tgt_dir, seq_to_str(opc.HABITUATED_SEQUENCE_IMAGES))
np.save(save_file_name, habituated_sequence_data)


# # Generate randomized oddballs block:
habituation_oddball_full_sequence = get_shuffled_repeated_sequence(opc.HABITUATED_ODDBALL_IMAGES, 100, seed=0)
randomized_oddballs_data = generate_sequence_block(habituation_oddball_full_sequence, src_image_data)
save_file_name = os.path.join(tgt_dir, 'HABITUATION_RANDOMIZED_ODDBALL.npy')
np.save(save_file_name, randomized_oddballs_data)
