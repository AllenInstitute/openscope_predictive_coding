"""
OpenScope Habituation Script
"""
import os
from psychopy import visual
from camstim import SweepStim, MovieStim
from camstim import Foraging
from camstim import Window
import numpy as np
import hashlib


data_path = r'//allen/aibs/technology/nicholasc/openscope'


n_repeats = 2
expected_gray_screen_duration = 60.0
expected_randomized_oddball_duration = 250.0*n_repeats
expected_habituated_sequence_duration = 100.0*n_repeats
expected_familiar_movie_duration = 150.0*n_repeats
expected_total_duration = expected_familiar_movie_duration+expected_habituated_sequence_duration+expected_randomized_oddball_duration+3*expected_gray_screen_duration

assert os.path.basename(__file__).split('.')[0][-1] == str(n_repeats)


# Create display window, warped
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',
                screen=0)

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
file_name = os.path.join(data_path, 'habituation_pilot_randomized_oddball_5cd9854e9cb07a427180d6e130c148ab.npy')
data = np.load(file_name)
number_of_frames = data.shape[0]
runs = 1*n_repeats
frame_length = .25
timing_list = [(ii*frame_length*number_of_frames, (ii+1)*frame_length*number_of_frames) for ii in range(runs)]
curr_stimulus_list, tf = get_block(file_name, timing_list, frame_length, runs, t0=t0, )
stimuli.append(curr_stimulus_list)
assert tf - t0 == expected_randomized_oddball_duration

# Spontaneous gray screen block 2:
tf += 60

# habituated sequence block:
t0 = tf
file_name = os.path.join(data_path, '68_78_13_26_2950e8d1e5187ce65ac40f5381be0b3f.npy')
data = np.load(file_name)
number_of_frames = data.shape[0]
runs = 20*5*n_repeats
frame_length = .25
timing_list = [(ii*frame_length*number_of_frames, (ii+1)*frame_length*number_of_frames) for ii in range(runs)]
curr_stimulus_list, tf = get_block(file_name, timing_list, frame_length, runs, t0=t0, )
stimuli.append(curr_stimulus_list)
assert tf - t0 == expected_habituated_sequence_duration


# Spontaneous gray screen block 3:
tf += 60

# Familiar movie block:
t0 = tf
file_name = os.path.join(data_path, 'natural_movie_one_warped_77ee4ecd0dc856c80cf24621303dd080.npy')
data = np.load(file_name)
number_of_frames = data.shape[0]
runs = 5*n_repeats
frame_length = 2.0/60.0
timing_list = [(ii*frame_length*number_of_frames, (ii+1)*frame_length*number_of_frames) for ii in range(runs)]
curr_stimulus_list, tf = get_block(file_name, timing_list, frame_length, runs, t0=t0)
stimuli.append(curr_stimulus_list)
assert tf - t0 == expected_familiar_movie_duration


assert tf == expected_total_duration
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
