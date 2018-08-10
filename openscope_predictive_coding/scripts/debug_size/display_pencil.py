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
import json

data_path = r'//allen/aibs/technology/nicholasc/openscope'


checksum_dict = json.load(open(os.path.join(data_path, 'stimulus_pilot_checksum_dict_%s.json' % 'B'), 'r'))


if __name__ == "__main__":

    # Create display window, warped
    window = Window(fullscr=True,
                    monitor='Gamma1.Luminance50',
                    screen=0,
                    warp=Warp.Spherical
                    )

    def get_pencil_stimulus(frame_length=1., t0=0):
        movie_path = os.path.join(data_path, '115_114.npy')
        movie_data = np.load(movie_path)

        base_seq_stim = MovieStim(movie_path=movie_path,
                                        window=window,
                                        frame_length=frame_length,
                                        size=(1920, 1080),
                                        start_time=0.0,
                                        stop_time=None,
                                        flip_v=True,
                                        runs=10,)


        return [base_seq_stim]


    # Spontaneous gray screen block (offset):
    pencil_stimulus_list = get_pencil_stimulus(frame_length=2., t0=0)

    

    stimuli = pencil_stimulus_list

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
