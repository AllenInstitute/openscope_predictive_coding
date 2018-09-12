import os
import numpy as np

data_path = os.path.join(os.path.dirname(__file__), 'data')
stimtable_path = os.path.join(os.path.dirname(__file__), 'scripts', 'ophys')
boc_path = '/home/nicholasc/boc'

SEQUENCE_IMAGES = {}
SEQUENCE_IMAGES['A'] = (68, 78, 13, 26)
SEQUENCE_IMAGES['B'] = (53, 44, 8, 14)
SEQUENCE_IMAGES['C'] = (114, 23, 65, 101)

ODDBALL_IMAGES = {}
ODDBALL_IMAGES['A'] = (6, 22, 51, 71, 111, 17, 110, 112, 103, 89)
ODDBALL_IMAGES['B'] = (7, 94, 9, 29, 117, 42, 10, 45, 41, 19)
ODDBALL_IMAGES['C'] = (96, 84, 32, 15, 115, 27, 40, 52, 93, 35)


HABITUATED_SEQUENCE_IMAGES = SEQUENCE_IMAGES['A']
HABITUATED_ODDBALL_IMAGES = ODDBALL_IMAGES['A']

IMAGE_W = 960
IMAGE_H = 600

SCREEN_W = 1920
SCREEN_H = 1200

assert SCREEN_W == IMAGE_W*2 and SCREEN_H == IMAGE_H*2

from .utilities import memoized

# @memoized
# def get_dataset_template(key):
#     curr_path = os.path.join(data_path, 'templates', '%s.npy' % key)
#     return np.load(curr_path)

@memoized
def get_dataset(key):
    curr_path = os.path.join(data_path, '%s.npy' % key)
    return np.load(curr_path)

OCCLUSION_DOT_SIZE = 48
OCCLUSION_NUM_DOT_LIST = [0,   20,   44,   66,  100,  136]
OCCLUSION_NUM_DOT_to_FRACTION = {0:0., 20:.15, 44:.3, 66:.4, 100:.5, 136:.6}

# Consistency check:
for key in OCCLUSION_NUM_DOT_to_FRACTION:
    assert key in OCCLUSION_NUM_DOT_LIST

import stimulus