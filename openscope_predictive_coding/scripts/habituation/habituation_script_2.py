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
expected_total_duration = expected_familiar_movie_duration+expected_habituated_sequence_duration+expected_randomized_oddball_duration+4*expected_gray_screen_duration

assert os.path.basename(__file__).split('.')[0][-1] == str(n_repeats)


def seq_to_str(sequence):
    return '_'.join([str(ii) for ii in sequence])

def get_hash(data):
    return hashlib.md5(data).hexdigest()

HABITUATED_SEQUENCE_IMAGES = [68, 78, 13, 26]


# Create display window, warped
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',
                screen=0)

def get_block(file_name, timing_list, frame_length, runs, t0):

        movie_data = np.load(file_name)
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

# def get_sequence_block(base_seq, cycle_length, frame_length=.25, t0=0):
#     base_seq_str = '_'.join([str(ii) for ii in base_seq])
#     movie_path = os.path.join(data_path, '%s.npy' % base_seq_str)
#     movie_data = np.load(movie_path)
#     assert hashlib.md5(movie_data).hexdigest() == checksum_dict[movie_path.replace('\\','/')]

#     base_seq_stim = MovieStim(movie_path=movie_path,
#                                     window=window,
#                                     frame_length=frame_length,
#                                     size=(1920, 1080),
#                                     start_time=0.0,
#                                     stop_time=None,
#                                     flip_v=True,
#                                     runs=cycle_length,)
#     t0_b, tf_b = t0, t0 + len(base_seq)*frame_length*cycle_length
#     base_seq_stim.set_display_sequence([(t0_b, tf_b)])
#     return [base_seq_stim], tf_b

# def get_natural_movie_block(cycle_length, frame_length=2.0/60.0, t0=0):
#     movie_path = os.path.join(data_path, 'NATURAL_MOVIE_ONE.npy')
#     movie_data = np.load(movie_path)
#     assert hashlib.md5(movie_data).hexdigest() == checksum_dict[movie_path.replace('\\','/')]
    
#     movie_duration, movie_width, movie_height = movie_data.shape
#     movie_stim = MovieStim(movie_path=movie_path,
#                                     window=window,
#                                     frame_length=frame_length,
#                                     size=(1920, 1200),
#                                     start_time=0.0,
#                                     stop_time=None,
#                                     flip_v=True,
#                                     runs=cycle_length,)
    
#     t0_b, tf_b = t0, t0 + movie_duration*frame_length*cycle_length
#     movie_stim.set_display_sequence([(t0_b, tf_b)])
#     return [movie_stim], tf_b

# def get_randomized_oddball_image_block(cycle_length, frame_length=.25, t0=0):
#     image_path = os.path.join(data_path, 'habituation_randomized_oddball.npy')
#     image_data = np.load(image_path)
#     assert hashlib.md5(image_data).hexdigest() == checksum_dict[image_path.replace('\\','/')]

#     number_of_images, image_width, image_height = image_data.shape
#     image_stim = MovieStim(movie_path=image_path,
#                                     window=window,
#                                     frame_length=frame_length,
#                                     size=(1920, 1200),
#                                     start_time=0.0,
#                                     stop_time=None,
#                                     flip_v=True,
#                                     runs=cycle_length,)

#     t0_b, tf_b = t0, t0 + number_of_images*frame_length*cycle_length
#     image_stim.set_display_sequence([(t0_b, tf_b)])
#     return [image_stim], tf_b


# def get_spontaneous_gray_screen_block(duration, t0=0):
#     # This keeps the function signatures below looking similar, to prevent typos

#     return [], t0 + duration

# Initialize:
tf = 0
stimuli = []

# Spontaneous gray screen block 1:
tf += 60

# Randomized oddball block:
t0 = tf
file_name = os.path.join(data_path, 'sequences', 'HABITUATION_RANDOMIZED_ODDBALL.npy')
data = np.load(file_name)
assert get_hash(data) == '5cd9854e9cb07a427180d6e130c148ab'
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
file_name = os.path.join(data_path, 'sequences', '%s.npy' % seq_to_str(HABITUATED_SEQUENCE_IMAGES))
data = np.load(file_name)
assert get_hash(data) == '2950e8d1e5187ce65ac40f5381be0b3f'
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
file_name = os.path.join(data_path, 'templates', 'NATURAL_MOVIE_ONE_WARPED.npy')
data = np.load(file_name)
assert get_hash(data) == '77ee4ecd0dc856c80cf24621303dd080'
number_of_frames = data.shape[0]
runs = 5*n_repeats
frame_length = 2.0/60.0
timing_list = [(ii*frame_length*number_of_frames, (ii+1)*frame_length*number_of_frames) for ii in range(runs)]
curr_stimulus_list, tf = get_block(file_name, timing_list, frame_length, runs, t0=t0)
stimuli.append(curr_stimulus_list)
assert tf - t0 == expected_familiar_movie_duration

# # Spontaneous gray screen block 4:
tf += 60


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
