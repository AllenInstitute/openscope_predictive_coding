"""
OpenScope Habituation Script
"""
import os
from psychopy import visual
from camstim import Stimulus, SweepStim, MovieStim, NaturalScenes
from camstim import Foraging
from camstim import Window, Warp
import glob
import numpy as np

data_path = r'//allen/aibs/technology/nicholasc/openscope'

# Create display window, warped
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',
                screen=0,
                warp=Warp.Spherical
                )


def get_sequence_block(base_seq, cycle_length, frame_length=.25, t0=0):
    movie_path = os.path.join(data_path, '%s.npy' % base_seq)
    base_seq_stim = MovieStim(movie_path=movie_path,
                                       window=window,
                                       frame_length=frame_length,
                                       size=(1920, 1080),
                                       start_time=0.0,
                                       stop_time=None,
                                       flip_v=True,
                                       runs=cycle_length,)
    t0_b, tf_b = t0, t0 + len(base_seq)*frame_length*(cycle_length-2)
    base_seq_stim.set_display_sequence([(t0_b, tf_b)])
    return [base_seq_stim], tf_b

def get_natural_movie_block(cycle_length, frame_length=2.0/60.0, t0=0):
    movie_path = os.path.join(data_path, 'NATURAL_MOVIE_ONE.npy')
    movie_data = np.load(movie_path)
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

def get_randomized_oddball_image_block(cycle_length, frame_length=.25, t0=0):
    image_path = os.path.join(data_path, 'NATURAL_SCENES_LUMINANCE_MATCHED.npy')
    image_data = np.load(image_path)
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

s_seq, tf = get_sequence_block('ABCD', 20*5, t0=0)
s_movie, tf = get_natural_movie_block(5, t0=0)
s_natural, tf = get_randomized_oddball_image_block(1, t0=0)
stimuli = s_movie



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
