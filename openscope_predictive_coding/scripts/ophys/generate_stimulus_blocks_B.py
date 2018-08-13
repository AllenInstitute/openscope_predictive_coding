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
session_type = 'B'

assert os.path.basename(__file__).split('.')[0][-1] == session_type

SEQUENCE_IMAGES = opc.SEQUENCE_IMAGES[session_type]
ODDBALL_IMAGES = opc.ODDBALL_IMAGES[session_type]

oddball_checksum_list = ["ab5644ce065d12d4bef02982048c1162",
                        "df1a9ba73991af88df190aa0796fcd86",
                        "8482e756f08df363a615e17f022d7076",
                        "32aed2bc5175f77ab70ea0012eefc397",
                        "97412d675a020b77dfe6a9c132be92c6",
                        "9cbe99ec99a9d5dccae6d336c2d78ffe",
                        "6126b9b6fb81a2d99a69c53800922390",
                        "25f32c6b10d409ad93f29f4370a98a67",
                        "4d8c88a783c27b69072b41946646b9f3",
                        "973259b77ae319e9ac6ae3e5da57e95b",]

pair_checksum_list = ["f0bb86647603d47fae185eda34afed80",
                    "8e2081daa42e6538efa99d1e50122e82",
                    "4dec9bce75d8be13c52ff0dd7b809ce7",
                    "43711bbe958bb58a3bc22e6ae3d02a1e",
                    "1406691ac248e0cf22f32e95cd99325f",
                    "4cfdb5e2e4e6db6fb2a69507130c6199",
                    "dae3efa675663ef439d8f8a2bef63677",
                    "0cec8e5f5618844ace55555bc0266294",
                    "d43cdde8a3033dc6c4a6f334523d6bd2",
                    "9b6fcfe22ed0a681ffd6ad8c6783d367",
                    "1bb4f79f4f3a814125e3d7a8f0bce959",
                    "9af2e373414d0b5ac9c3d915252a7a0f",
                    "2bee87b5e737b4458f059eb2879978a0",
                    "f810a847ae64785c67a79a909be5bd09",
                    "8fed3bef24891d63c797c9f62668588c",
                    "2322a219fd4e90500b2d85d3a55a1a7f",
                    "382e4f85298acf0df6d43d8ec8450ca0",
                    "7bd0e82a2d4e8a3945eba86af752d26c",
                    "233c55e08dbcf476ab4848fb1428051a",
                    "ea1db11b1766c0c2ebb1ef14eb12e3ec",
                    "3c3e2e89497624b84a50a6b0a4f1d243",
                    "9d22fcbec1e2e2f73f6b4883802fbbc6",
                    "938a48e30d55be2a7dd9e1589ffe1857",
                    "ee24a4bacefe0cbfe2d3610dc094f6b1",]

stimulus_pilot_checksum_dict = {}
stimulus_pilot_data = {}

# Generate base sequence:
save_file_name = os.path.join(tgt_dir, '%s.npy' % '_'.join([str(x) for x in SEQUENCE_IMAGES]))
base_block_checksum, base_block_file_name = utilities.generate_sequence_block(SEQUENCE_IMAGES, save_file_name=save_file_name)
assert base_block_checksum == '41e5f2b7f0fad6735a4eb566514efd1c'
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
assert hab_randomized_control_checksum == '2c3fd3b6529df6160df336a16fdc9862'
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







