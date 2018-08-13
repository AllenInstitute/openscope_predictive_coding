import random
import os
import numpy as np
import hashlib
import copy
from collections import defaultdict

src_file_name = '/allen/aibs/technology/nicholasc/openscope/NATURAL_SCENES_LUMINANCE_MATCHED.npy'
src_image_data = np.load(src_file_name)

def get_hash(data):

    return hashlib.md5(data).hexdigest()

def get_shuffled_repeated_sequence(src_sequence, number_of_shuffles, seed=None):
    
    if seed is not None:
        random.seed(seed)
    
    new_sequence = []
    for x in range(number_of_shuffles):
        random.shuffle(src_sequence)
        new_sequence += list(src_sequence)

    return new_sequence

def generate_oddball_block_timing_dict(base_seq, oddball_list, num_cycles_per_repeat=20, oddball_cycle_min=10, oddball_cycle_max=19, num_repeats_per_oddball=10, frame_length=.25, expected_duration=None, seed=None):
    
    # Force to be zero, to avoid confusion when making script
    t0=0

    if seed is not None:
        random.seed(seed)

    if not expected_duration is None:
        np.testing.assert_approx_equal(expected_duration, len(base_seq)*len(oddball_list)*num_cycles_per_repeat*num_repeats_per_oddball*frame_length)
    
    delta_t = frame_length*len(base_seq)
    t = t0
    timing_dict = defaultdict(list)
    for _ in range(num_repeats_per_oddball):
        
        oddball_list_shuffle = [x for x in oddball_list]
        random.shuffle(oddball_list_shuffle)

        for oddball in oddball_list_shuffle:
            
            cycle_to_replace = random.choice(range(oddball_cycle_min-1,oddball_cycle_max)) 
            curr_repeat = [[x for x in base_seq] for _ in range(num_cycles_per_repeat)]
            curr_repeat[cycle_to_replace][-1] = oddball

            for curr_cycle in curr_repeat:
                timing_dict[tuple(curr_cycle)].append((t, t+delta_t))
                t += delta_t            


    # Lots of Double-checking:
    number_of_total_cycles = 0
    timing_hwm = -float('inf')
    for pattern, timing_list in timing_dict.items():
        number_of_total_cycles += len(timing_list)

        if pattern != tuple(base_seq):
            assert len(timing_list) == num_repeats_per_oddball
        for timing_tuple in timing_list:
            assert timing_tuple[0] < timing_tuple[1]
            np.testing.assert_approx_equal( (timing_tuple[1] - timing_tuple[0])/len(pattern), frame_length)
            if timing_tuple[1] > timing_hwm:
                timing_hwm = timing_tuple[1]
    assert number_of_total_cycles == len(oddball_list)*num_cycles_per_repeat*num_repeats_per_oddball
    assert len(timing_dict[tuple(base_seq)]) == number_of_total_cycles - num_repeats_per_oddball*len(oddball_list)

    np.testing.assert_approx_equal(t0+len(base_seq)*len(oddball_list)*num_cycles_per_repeat*num_repeats_per_oddball*frame_length, timing_hwm)

    return timing_dict

def generate_sequence_block(base_seq, save_file_name):
    
    base_seq_str = '_'.join([str(ii) for ii in base_seq])

    N = len(base_seq)
    h, w = src_image_data.shape[1:]
    data_block = np.zeros((N,h,w), dtype=np.uint8)
    for ii, idx in enumerate(base_seq):
        data_block[ii,:,:] = src_image_data[idx,:,:]

    np.save(save_file_name, data_block)
    return hashlib.md5(data_block).hexdigest(), save_file_name

def generate_pair_block_timing_dict(pair_list, num_repeats=30, frame_length=.25, expected_duration=None, seed=None):

    if seed is not None:
        random.seed(seed)

    for x in pair_list:
        assert len(x) == 2

    timing_list = []
    t = 0.
    delta_t = t + frame_length*2
    for ii in range(len(pair_list)*num_repeats):
        timing_list.append((t, t+delta_t))
        t += delta_t
    
    timing_dict = defaultdict(list)
    for ii in range(num_repeats):
        curr_repeat_time_list = timing_list[ii*len(pair_list):(ii+1)*len(pair_list)]
        tmp_pair_list = [x for x in pair_list]
        random.shuffle(tmp_pair_list)
        for curr_pair, curr_repeat_time in zip(tmp_pair_list, curr_repeat_time_list):
            timing_dict[curr_pair].append(curr_repeat_time)
    
    for curr_timing_list in timing_dict.values():
        curr_timing_list.sort(key=lambda x:x[0])
    
    return timing_dict
