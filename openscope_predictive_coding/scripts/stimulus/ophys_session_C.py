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
session_type = 'C'

# Consistency check:
assert os.path.basename(__file__).split('.')[0][-1] == session_type

checksum_dict = json.load(open(os.path.join(data_path, 'stimulus_pilot_checksum_dict_%s.json' % session_type), 'r'))

if __name__ == "__main__":

    # Create display window, warped
    window = Window(fullscr=True,
                    monitor='Gamma1.Luminance50',
                    screen=0,
                    warp=Warp.Spherical
                    )

    def get_sequence_block(base_seq, timing_list, t0=0):
        cycle_length=len(base_seq)
        number_of_repeats=len(timing_list)

        frame_length = (timing_list[0][1]-timing_list[0][0])/float(cycle_length)
        base_seq_str = '_'.join([str(ii) for ii in base_seq])
        movie_path = os.path.join(data_path, '%s.npy' % base_seq_str)
        movie_data = np.load(movie_path)
        assert hashlib.md5(movie_data).hexdigest() == checksum_dict[movie_path.replace('\\','/')]

        base_seq_stim = MovieStim(movie_path=movie_path,
                                        window=window,
                                        frame_length=frame_length,
                                        size=(1920, 1080),
                                        start_time=0.0,
                                        stop_time=None,
                                        flip_v=True,
                                        runs=number_of_repeats,)

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
        return [base_seq_stim], timing_hwm

    def get_natural_movie_block(cycle_length, movie_name, frame_length=2.0/60.0, t0=0):
        movie_path = os.path.join(data_path, '%s.npy' % movie_name)
        movie_data = np.load(movie_path)
        assert hashlib.md5(movie_data).hexdigest() == checksum_dict[movie_path.replace('\\','/')]
        
        movie_duration, movie_width, movie_height = movie_data.shape
        movie_stim = MovieStim(movie_path=movie_path,
                                        window=window,
                                        frame_length=frame_length,
                                        size=(1920, 1200),
                                        start_time=0.0,
                                        stop_time=None,
                                        flip_v=True,
                                        runs=cycle_length,)
        
        t0_b, tf_b = t0, t0 + movie_duration*frame_length*cycle_length
        movie_stim.set_display_sequence([(t0_b, tf_b)])
        return [movie_stim], tf_b

    def get_randomized_control(cycle_length, frame_length=.25, t0=0):
        image_path = os.path.join(data_path, 'randomized_control_%s.npy' % session_type)
        image_data = np.load(image_path)
        assert hashlib.md5(image_data).hexdigest() == checksum_dict[image_path.replace('\\','/')]

        number_of_images, image_width, image_height = image_data.shape
        image_stim = MovieStim(movie_path=image_path,
                                        window=window,
                                        frame_length=frame_length,
                                        size=(1920, 1200),
                                        start_time=0.0,
                                        stop_time=None,
                                        flip_v=True,
                                        runs=cycle_length,)

        t0_b, tf_b = t0, t0 + number_of_images*frame_length*cycle_length
        image_stim.set_display_sequence([(t0_b, tf_b)])
        return [image_stim], tf_b


    def get_spontaneous_gray_screen_block(duration, t0=0):
        # This keeps the function signatures below looking similar, to prevent typos

        return [], t0 + duration

    # Initialize:
    tf = 0

    # # Spontaneous gray screen block (offset):
    t0 = tf
    spontaneous_gray_screen_stimulus_list_1, tf = get_spontaneous_gray_screen_block(60., t0=t0)
    assert tf - t0 == expected_gray_screen_duration

    # Randomized control pre:
    t0 = tf
    randomized_control_list_pre, tf = get_randomized_control(1, t0=t0)
    assert tf - t0 == expected_randomized_control_duration

    # Spontaneous gray screen block (offset):
    t0 = tf
    spontaneous_gray_screen_stimulus_list_2, tf = get_spontaneous_gray_screen_block(60., t0=t0)
    assert tf - t0 == expected_gray_screen_duration

    # oddball_stim_list sequence block:
    t0 = tf
    oddball_list = json.load(open(os.path.join(data_path, 'stimulus_pilot_data_%s.json' % session_type), 'r'))['oddball_timing']
    oddball_stimulus_list = []
    tf_list = []
    for pattern, timing_list in oddball_list:
        curr_stim, curr_tf = get_sequence_block(pattern, timing_list=timing_list, t0=t0)
        oddball_stimulus_list += curr_stim
        tf_list.append(curr_tf)
    tf = max(tf_list)
    assert tf - t0 == expected_oddball_stimulus_list_duration


    # Spontaneous gray screen block (offset):
    t0 = tf
    spontaneous_gray_screen_stimulus_list_3, tf = get_spontaneous_gray_screen_block(60., t0=t0)
    assert tf - t0 == expected_gray_screen_duration


    # Pair control sequence block:
    t0 = tf
    pair_list = json.load(open(os.path.join(data_path, 'stimulus_pilot_data_%s.json' % session_type), 'r'))['pair_timing']
    pair_stimulus_list = []
    tf_list = []
    for pattern, timing_list in pair_list:
        curr_stim, curr_tf = get_sequence_block(pattern, timing_list=timing_list, t0=t0)
        pair_stimulus_list += curr_stim
        tf_list.append(curr_tf)
    tf = max(tf_list)
    assert tf - t0 == expected_pair_control_duration


    # Spontaneous gray screen block (offset):
    t0 = tf
    spontaneous_gray_screen_stimulus_list_4, tf = get_spontaneous_gray_screen_block(60., t0=t0)
    assert tf - t0 == expected_gray_screen_duration


    # Familiar movie block:
    t0 = tf
    familiar_movie_stimulus_list, tf = get_natural_movie_block(10, 'NATURAL_MOVIE_ONE', t0=t0)
    assert tf - t0 == expected_familiar_movie_duration


    # Spontaneous gray screen block (offset):
    t0 = tf
    spontaneous_gray_screen_stimulus_list_5, tf = get_spontaneous_gray_screen_block(60., t0=t0)
    assert tf - t0 == expected_gray_screen_duration


    # Novel movie block:
    t0 = tf
    novel_movie_stimulus_list, tf = get_natural_movie_block(10, 'NATURAL_MOVIE_TWO', t0=t0)
    assert tf - t0 == expected_novel_movie_duration


    # Spontaneous gray screen block (offset):
    t0 = tf
    spontaneous_gray_screen_stimulus_list_6, tf = get_spontaneous_gray_screen_block(60., t0=t0)
    assert tf - t0 == expected_gray_screen_duration


    # Randomized control post:
    t0 = tf
    randomized_control_list_post, tf = get_randomized_control(1, t0=t0)
    assert tf - t0 == expected_randomized_control_duration


    # Spontaneous gray screen block (offset):
    t0 = tf
    spontaneous_gray_screen_stimulus_list_7, tf = get_spontaneous_gray_screen_block(60., t0=t0)
    assert tf - t0 == expected_gray_screen_duration
    assert tf == expected_total_duration

    stimuli = []
    stimuli += spontaneous_gray_screen_stimulus_list_1
    stimuli += randomized_control_list_pre
    stimuli += spontaneous_gray_screen_stimulus_list_2
    stimuli += oddball_stimulus_list
    stimuli += spontaneous_gray_screen_stimulus_list_3
    stimuli += pair_stimulus_list
    stimuli += spontaneous_gray_screen_stimulus_list_4
    stimuli += novel_movie_stimulus_list
    stimuli += spontaneous_gray_screen_stimulus_list_5
    stimuli += familiar_movie_stimulus_list
    stimuli += spontaneous_gray_screen_stimulus_list_6
    stimuli += randomized_control_list_post
    stimuli += spontaneous_gray_screen_stimulus_list_7

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
