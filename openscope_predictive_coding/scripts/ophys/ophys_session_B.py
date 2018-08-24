"""
OpenScope OPhys Stimulus Script
"""
import os
from psychopy import visual
from camstim import Stimulus, SweepStim, MovieStim, NaturalScenes
from camstim import Foraging
from camstim import Window, Warp
import glob
import numpy as np
import hashlib
import json
import warnings

data_path = r'//allen/aibs/technology/nicholasc/openscope'
expected_gray_screen_duration = 60.0
expected_randomized_control_duration = 105.0
expected_oddball_stimulus_list_duration = 2000.0
expected_pair_control_duration = 360.0
expected_familiar_movie_duration = 300.0
expected_novel_movie_duration = 300.0
expected_total_duration = 3590.0
session_type = 'B'

# Consistency check:
assert os.path.basename(__file__).split('.')[0][-1] == session_type


# Create display window, warped
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',
                screen=0,
                )

def get_block(file_name, timing_list, frame_length, runs, t0):
    
    base_seq_stim = MovieStim(movie_path=file_name,
                                    window=window,
                                    frame_length=frame_length,
                                    size=(1920, 1200),
                                    start_time=0.0,
                                    stop_time=None,
                                    flip_v=True,
                                    runs=runs,)

    # Shift t0:
    timing_hwm = -float('inf')
    timing_list_new = []
    for t_start, t_end in timing_list:
        t_start_new = t_start+t0
        t_end_new = t_end+t0
        timing_list_new.append((t_start_new, t_end_new))
        if t_end_new > timing_hwm:
            timing_hwm = t_end_new

    base_seq_stim.set_display_sequence(timing_list_new)
    return base_seq_stim, timing_hwm


# Initialize:
tf = 0
stimuli = []

# Spontaneous gray screen block 1:
tf += 60

# Randomized oddball block:
t0 = tf
file_name = os.path.join(data_path, 'ophys_pilot_randomized_control_B_1ce104b1011311ac984e647054fd253f.npy')
data = np.load(file_name)
number_of_frames = data.shape[0]
runs = 1
frame_length = .25
timing_list = [(ii*frame_length*number_of_frames, (ii+1)*frame_length*number_of_frames) for ii in range(runs)]
curr_stimulus_list, tf = get_block(file_name, timing_list, frame_length, runs, t0=t0, )
stimuli.append(curr_stimulus_list)
assert tf - t0 == expected_randomized_control_duration

# Spontaneous gray screen block 2:
tf += 60



# # # Spontaneous gray screen block (offset):
# t0 = tf
# spontaneous_gray_screen_stimulus_list_1, tf = get_spontaneous_gray_screen_block(60., t0=t0)
# assert tf - t0 == expected_gray_screen_duration

# # Randomized control pre:
# t0 = tf
# randomized_control_list_pre, tf = get_randomized_control(1, t0=t0)
# assert tf - t0 == expected_randomized_control_duration

# # Spontaneous gray screen block (offset):
# t0 = tf
# spontaneous_gray_screen_stimulus_list_2, tf = get_spontaneous_gray_screen_block(60., t0=t0)
# assert tf - t0 == expected_gray_screen_duration

# # oddball_stim_list sequence block:
# t0 = tf
# oddball_list = json.load(open(os.path.join(data_path, 'stimulus_pilot_data_%s.json' % session_type), 'r'))['oddball_timing']
# oddball_stimulus_list = []
# tf_list = []
# for pattern, timing_list in oddball_list:
#     curr_stim, curr_tf = get_sequence_block(pattern, timing_list=timing_list, t0=t0)
#     oddball_stimulus_list += curr_stim
#     tf_list.append(curr_tf)
# tf = max(tf_list)
# assert tf - t0 == expected_oddball_stimulus_list_duration


# # Spontaneous gray screen block (offset):
# t0 = tf
# spontaneous_gray_screen_stimulus_list_3, tf = get_spontaneous_gray_screen_block(60., t0=t0)
# assert tf - t0 == expected_gray_screen_duration


# # Pair control sequence block:
# t0 = tf
# pair_list = json.load(open(os.path.join(data_path, 'stimulus_pilot_data_%s.json' % session_type), 'r'))['pair_timing']
# pair_stimulus_list = []
# tf_list = []
# for pattern, timing_list in pair_list:
#     curr_stim, curr_tf = get_sequence_block(pattern, timing_list=timing_list, t0=t0)
#     pair_stimulus_list += curr_stim
#     tf_list.append(curr_tf)
# tf = max(tf_list)
# assert tf - t0 == expected_pair_control_duration


# # Spontaneous gray screen block (offset):
# t0 = tf
# spontaneous_gray_screen_stimulus_list_4, tf = get_spontaneous_gray_screen_block(60., t0=t0)
# assert tf - t0 == expected_gray_screen_duration


# # Familiar movie block:
# t0 = tf
# familiar_movie_stimulus_list, tf = get_natural_movie_block(10, 'NATURAL_MOVIE_ONE', t0=t0)
# assert tf - t0 == expected_familiar_movie_duration


# # Spontaneous gray screen block (offset):
# t0 = tf
# spontaneous_gray_screen_stimulus_list_5, tf = get_spontaneous_gray_screen_block(60., t0=t0)
# assert tf - t0 == expected_gray_screen_duration


# # Novel movie block:
# t0 = tf
# novel_movie_stimulus_list, tf = get_natural_movie_block(10, 'NATURAL_MOVIE_TWO', t0=t0)
# assert tf - t0 == expected_novel_movie_duration


# # Spontaneous gray screen block (offset):
# t0 = tf
# spontaneous_gray_screen_stimulus_list_6, tf = get_spontaneous_gray_screen_block(60., t0=t0)
# assert tf - t0 == expected_gray_screen_duration


# # Randomized control post:
# t0 = tf
# randomized_control_list_post, tf = get_randomized_control(1, t0=t0)
# assert tf - t0 == expected_randomized_control_duration


# # Spontaneous gray screen block (offset):
# t0 = tf
# spontaneous_gray_screen_stimulus_list_7, tf = get_spontaneous_gray_screen_block(60., t0=t0)
# assert tf - t0 == expected_gray_screen_duration
# assert tf == expected_total_duration



params = {}
ss = SweepStim(window,
            stimuli=stimuli,
            pre_blank_sec=0,
            post_blank_sec=0,
            params=params,
            )

f = Foraging(window=window,
            auto_update=False,
            params=params,
            nidaq_tasks={'digital_input': ss.di,
                        'digital_output': ss.do,})  #share di and do with SS
ss.add_item(f, "foraging")
ss.run()
