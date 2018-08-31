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

import collections
import pandas as pd
import time
import xxhash


data_path = r'//allen/aibs/technology/nicholasc/openscope'

import pyglet.gl as GL
n_repeats = 1
expected_gray_screen_duration = 60.0
expected_randomized_oddball_duration = 250.0*n_repeats
expected_habituated_sequence_duration = 100.0*n_repeats
expected_familiar_movie_duration = 150.0*n_repeats
expected_total_duration = expected_familiar_movie_duration+expected_habituated_sequence_duration+expected_randomized_oddball_duration+3*expected_gray_screen_duration

assert os.path.basename(__file__).split('.')[0][-1] == str(n_repeats)

def add_frame_dir(sweep_stim, output_dir='.'):

    image_dict = {}
    timing_dict = collections.defaultdict(list)

    for stimulus in sweep_stim.stimuli:
        old_update = stimulus.update
        def new_update(frame):
            old_update(frame)
            t0 = time.time()
            # print 
            # assert frame == stimulus.current_frame
            # print 'A1', t0 - time.time()
            # sweep_stim.window.getMovieFrame()

            w, h = 1920, 1200
            left = top = 0
            bufferDat = (GL.GLubyte * (4 * w * h))()                
            GL.glReadPixels(left, top, w, h,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, bufferDat)
            # print GL
            # print 'A2', t0 - time.time()
            # curr_image = sweep_stim.window.movieFrames.pop()
            # print 'A3', t0 - time.time()
            # tmp = curr_image.resize((curr_image.size[0]/8, curr_image.size[1]/8))
            # print 'A4', t0 - time.time()
            # t0 = time.time()
            # curr_frame = np.array(tmp)
            # print 'B', t0 - time.time()
            # t0 = time.time()
            # curr_frame_hash = xxhash.xxh64(curr_frame).digest()

            # if curr_frame_hash not in image_dict:
            #     image_dict[curr_frame_hash] = curr_image
            # timing_dict['timestamp'].append(time.time()-sweep_stim.start_time)
            # timing_dict['xxhash'].append(curr_frame_hash)
            print 'C', time.time()-t0 

        stimulus.update = new_update

    old_finalize = sweep_stim._finalize

    
    def new_finalize(*args, **kwargs):
        hash_filename_dict = collections.defaultdict(list)
        for key, val in image_dict.items():
            curr_array = np.array(val)
            curr_frame_md5 = hashlib.md5(curr_array).hexdigest()
            save_file_name = os.path.abspath(os.path.join(output_dir, '%s.npy' % curr_frame_md5))
            np.save(save_file_name, curr_array)

            hash_filename_dict['file_name'].append(save_file_name)

            curr_frame_downsample = np.array(val.resize((val.size[0]/8, val.size[1]/8)))
            curr_frame_xxhash = xxhash.xxh64(curr_frame_downsample).digest()
            hash_filename_dict['xxhash'].append(curr_frame_xxhash)

        
        timing_df = pd.DataFrame(timing_dict)
        timing_df = pd.merge(timing_df, pd.DataFrame(hash_filename_dict), on='xxhash').drop('xxhash', axis=1)

        timing_df.to_csv(os.path.join(output_dir, 'stimtable.csv'))
        old_finalize(*args, **kwargs)
    sweep_stim._finalize = new_finalize


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
# tf += 60

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

# # Spontaneous gray screen block 2:
# tf += 60

# # habituated sequence block:
# t0 = tf
# file_name = os.path.join(data_path, '68_78_13_26_2950e8d1e5187ce65ac40f5381be0b3f.npy')
# data = np.load(file_name)
# number_of_frames = data.shape[0]
# runs = 20*5*n_repeats
# frame_length = .25
# timing_list = [(ii*frame_length*number_of_frames, (ii+1)*frame_length*number_of_frames) for ii in range(runs)]
# curr_stimulus_list, tf = get_block(file_name, timing_list, frame_length, runs, t0=t0, )
# stimuli.append(curr_stimulus_list)
# assert tf - t0 == expected_habituated_sequence_duration


# # Spontaneous gray screen block 3:
# tf += 60

# # Familiar movie block:
# t0 = tf
# file_name = os.path.join(data_path, 'natural_movie_one_warped_77ee4ecd0dc856c80cf24621303dd080.npy')
# data = np.load(file_name)
# number_of_frames = data.shape[0]
# runs = 5*n_repeats
# frame_length = 2.0/60.0
# timing_list = [(ii*frame_length*number_of_frames, (ii+1)*frame_length*number_of_frames) for ii in range(runs)]
# curr_stimulus_list, tf = get_block(file_name, timing_list, frame_length, runs, t0=t0)
# stimuli.append(curr_stimulus_list)
# assert tf - t0 == expected_familiar_movie_duration


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

add_frame_dir(ss, output_dir='.')
ss.run()
