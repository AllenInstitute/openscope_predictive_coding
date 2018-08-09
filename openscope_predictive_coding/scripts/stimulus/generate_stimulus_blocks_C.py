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
session_type = 'C'

assert os.path.basename(__file__).split('.')[0][-1] == session_type

SEQUENCE_IMAGES = opc.SEQUENCE_IMAGES[session_type]
ODDBALL_IMAGES = opc.ODDBALL_IMAGES[session_type]

oddball_checksum_list = ["9aad9dc478dfd933589d7e675ffd1bc6",
                        "8883d50688f5838268f5659d9d561536",
                        "fa437b8a3cb0ecee4cfc8eb4af119266",
                        "1062d613b416bf2289edf44ea7914027",
                        "e15bd9c2a0cdbd20c9d9c42f84a8965b",
                        "f1ca73a20244aa613648a7c81d4225df",
                        "7358eb6ad40c8d65b1870029a9d664dc",
                        "577ec667c244f1f2adc77cb5daeb7d08",
                        "05eeabb07c045c1563435dbf33ec4683",
                        "71ed8e7b66fcef0d9ba22acf31cbd4d3",]

pair_checksum_list = ["581624fcf6f214e9d5f47a4f86859fc6",
                    "64370d3a980ab6445825d38ae2cb24f2",
                    "7fafd4148a805c7c08e1229b660fb313",
                    "62478a2bfc90ca22b2b1b0d65c15f6a5",
                    "70dfd32e2c1837208013981b163bf1d7",
                    "71738dda424e7ef7e7692d1e34dc45ad",
                    "e31d7f67260c11515ce8ef6decfbca75",
                    "fb573cfda4cc9e47620f8e068f636fd8",
                    "44b9b8f4bba4562b5fb52263d9752612",
                    "5aa526991c7c56fba8d8c499cfd2804f",
                    "3fc3d6394d165d95f362566ff795af70",
                    "f1caca595db966f7ff58256bbd853fdf",
                    "d5849107b7b29d47ddcf02e4545d3c64",
                    "4caa94e5652f137dd2df0b5cc54b5c8f",
                    "63d4884598b1ca49af659a1eac33e021",
                    "063c434809673a20a51b81ef7ab49494",
                    "8809b4a15aa8ffbefd7640f726591554",
                    "7b53275d0c005b88e175315609dd678c",
                    "5fbd94283f8a2ab33140220562a4efa0",
                    "db7c1d2781327ef7c43652eef7038731",
                    "a33907fb72d03f86c49d3a7254fac72e",
                    "788e49aac95ce6e265ae75a95b8abf11",
                    "7aaa63473e63970d8facf822000aad22",
                    "9dbb58f83d3964fba3fb0d8c2e2b9053",]

stimulus_pilot_checksum_dict = {}
stimulus_pilot_data = {}

# Generate base sequence:
save_file_name = os.path.join(tgt_dir, '%s.npy' % '_'.join([str(x) for x in SEQUENCE_IMAGES]))
base_block_checksum, base_block_file_name = utilities.generate_sequence_block(SEQUENCE_IMAGES, save_file_name=save_file_name)
assert base_block_checksum == '4cc6cf0c0b4abd23b60e79540b9175a4'
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
    pair_list.append((SEQUENCE_IMAGES[ii], SEQUENCE_IMAGES[jj]))
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
assert hab_randomized_control_checksum == '3c71dd8b15d716eade27c4a2a394fa2e'
stimulus_pilot_checksum_dict[hab_randomized_control_file_name] = hab_randomized_control_checksum


# Also add checksum for natural movie 1 from Brain Observatory:
stimulus_pilot_checksum_dict[os.path.join(tgt_dir, 'NATURAL_MOVIE_ONE.npy')] = 'b174ad09736c870c6915baf82cf2c9ad'
stimulus_pilot_checksum_dict[os.path.join(tgt_dir, 'NATURAL_MOVIE_TWO.npy')] = '68e5976a140fe8400c6b7fe59073fe72'


# Oddball data:
oddball_list = []
oddball_dict = utilities.generate_oddball_block_timing_dict(SEQUENCE_IMAGES, ODDBALL_IMAGES, expected_duration=2000.0, seed=0)
for key_val in oddball_dict.items():
    oddball_list.append(key_val)
stimulus_pilot_data['oddball_timing'] = oddball_list


# Dump to pilot data directory:
json.dump(stimulus_pilot_checksum_dict, open(os.path.join(tgt_dir, 'stimulus_pilot_checksum_dict_%s.json' % session_type), 'w'))
json.dump(stimulus_pilot_data, open(os.path.join(tgt_dir, 'stimulus_pilot_data_%s.json' % session_type), 'w'), indent=2)







