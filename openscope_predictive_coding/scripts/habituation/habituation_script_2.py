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
import hashlib

data_path = r'//allen/aibs/technology/nicholasc/openscope'
n_repeats = 2
expected_gray_screen_duration = 60.0
expected_randomized_oddball_duration = 250.0*n_repeats
expected_habituated_sequence_duration = 100.0*n_repeats
expected_familiar_movie_duration = 150.0*n_repeats
expected_total_duration = expected_familiar_movie_duration+expected_habituated_sequence_duration+expected_randomized_oddball_duration+2*expected_gray_screen_duration

HABITUATED_SEQUENCE_IMAGES = [68, 78, 13, 26]

if __name__ == "__main__":

    # Create display window, warped
    window = Window(fullscr=True,
                    monitor='Gamma1.Luminance50',
                    screen=0,
                    warp=Warp.Spherical
                    )


    def get_sequence_block(base_seq, cycle_length, frame_length=.25, t0=0):
        base_seq_str = '_'.join([str(ii) for ii in base_seq])
        movie_path = os.path.join(data_path, '%s.npy' % base_seq_str)
        movie_data = np.load(movie_path)
        assert hashlib.md5(movie_data).hexdigest() == 'd1de3dfc46a972d323c7b52b9fff78a9'

        base_seq_stim = MovieStim(movie_path=movie_path,
                                        window=window,
                                        frame_length=frame_length,
                                        size=(1920, 1080),
                                        start_time=0.0,
                                        stop_time=None,
                                        flip_v=True,
                                        runs=cycle_length,)
        t0_b, tf_b = t0, t0 + len(base_seq)*frame_length*cycle_length
        base_seq_stim.set_display_sequence([(t0_b, tf_b)])
        return [base_seq_stim], tf_b

    def get_natural_movie_block(cycle_length, frame_length=2.0/60.0, t0=0):
        movie_path = os.path.join(data_path, 'NATURAL_MOVIE_ONE.npy')
        movie_data = np.load(movie_path)
        assert hashlib.md5(movie_data).hexdigest() == 'b174ad09736c870c6915baf82cf2c9ad'
        
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
        image_path = os.path.join(data_path, 'habituation_randomized_oddball.npy')
        image_data = np.load(image_path)
        assert hashlib.md5(image_data).hexdigest() == 'c38555394253b83f42e2a257e1830c20'

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

    # Randomized oddballs block:
    t0 = 0
    randomized_oddball_stimulus_list, tf = get_randomized_oddball_image_block(1*n_repeats, t0=t0)
    assert tf - t0 == expected_randomized_oddball_duration

    # Spontaneous gray screen block (offset):
    t0 = tf
    spontaneous_gray_screen_stimulus_list_1, tf = get_spontaneous_gray_screen_block(60., t0=t0)
    assert tf - t0 == expected_gray_screen_duration

    # Habituated sequence block:
    t0 = tf
    habituated_sequence_stimulus_list, tf = get_sequence_block(HABITUATED_SEQUENCE_IMAGES, 20*5*n_repeats, t0=t0) # cycles-per-repeat times number of repeats
    assert tf - t0 == expected_habituated_sequence_duration

    # Spontaneous gray screen block (offset):
    t0 = tf
    spontaneous_gray_screen_stimulus_list_2, tf = get_spontaneous_gray_screen_block(60., t0=t0)
    assert tf - t0 == expected_gray_screen_duration

    # Familiar movie block:
    t0 = tf
    familiar_movie_stimulus_list, tf = get_natural_movie_block(5*n_repeats, t0=t0)
    assert tf - t0 == expected_familiar_movie_duration
    assert tf == expected_total_duration

    stimuli = randomized_oddball_stimulus_list + \
            spontaneous_gray_screen_stimulus_list_1 + \
            habituated_sequence_stimulus_list + \
            spontaneous_gray_screen_stimulus_list_2 + \
            familiar_movie_stimulus_list

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
