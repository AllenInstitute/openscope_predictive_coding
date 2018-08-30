import os
import numpy as np
import hashlib
import collections
import pandas as pd
import time
import xxhash

def add_frame_dir(sweep_stim, output_dir='.'):

    image_dict = {}
    timing_dict = collections.defaultdict(list)

    for stimulus in sweep_stim.stimuli:
        old_update = stimulus.update
        def new_update(frame):
            old_update(frame)
            assert frame == stimulus.current_frame

            sweep_stim.window.getMovieFrame()
            t0 = time.time()
            curr_image = sweep_stim.window.movieFrames.pop()
            tmp = curr_image.resize((curr_image.size[0]/8, curr_image.size[1]/8))
            curr_frame = np.array(tmp)
            curr_frame_hash = xxhash.xxh64(curr_frame).digest()

            if curr_frame_hash not in image_dict:
                image_dict[curr_frame_hash] = curr_image
            timing_dict['timestamp'].append(time.time()-sweep_stim.start_time)
            timing_dict['xxhash'].append(curr_frame_hash)

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

        
# ==============================================================================
# Minimal working example:
# ==============================================================================

from camstim import SweepStim, MovieStim
from camstim import Foraging
from camstim import Window


# Create display window, warped
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',
                screen=0)


movie_stim = MovieStim(movie_path='//allen/aibs/technology/nicholasc/openscope/22_68_e90bc7a3204e763dc7e295f87473dd38.npy',
                        window=window,
                        frame_length=1.,
                        size=(1920, 1200),
                        start_time=0.0,
                        stop_time=None,
                        flip_v=True,
                        runs=1,)

movie_stim.set_display_sequence([(1,3)])

params = {}
ss = SweepStim(window,
            stimuli=[movie_stim],
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
