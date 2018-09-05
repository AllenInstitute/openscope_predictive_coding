import os
import numpy as np
import hashlib
import collections
import pandas as pd
import time
import xxhash
from mss import mss
import ctypes

def add_frame_dir(sweep_stim, output_dir='.'):
    
    PLAINMASK = 0x00ffffff
    ZPIXMAP = 2
    width, height, top, left = 10, 10, 1200/2-10/2, 1920/2-10/2
    monitor = {'width': width, 'top': top, 'height': height, 'left': left}
    m = mss()
    
    image_dict = {}
    timing_dict = collections.defaultdict(list)

    for stimulus in sweep_stim.stimuli:
        old_update = stimulus.update
        def new_update(frame):
            t0 = time.time()
            old_update(frame)
            print time.time() - t0,
            t0 = time.time()
            assert frame == stimulus.current_frame
            ximage = m.xlib.XGetImage(
                        m.display,
                        m.drawable,
                        monitor["left"],
                        monitor["top"],
                        monitor["width"],
                        monitor["height"],
                        PLAINMASK,
                        ZPIXMAP,
                    )

            data = ctypes.cast(ximage.contents.data,
                            ctypes.POINTER(ctypes.c_ubyte * monitor["height"] * monitor["width"] * 4),)
            data_contents = bytearray(data.contents)
            m.xlib.XDestroyImage(ximage)
            curr_frame_hash = xxhash.xxh64(data_contents).digest()
            
            if curr_frame_hash not in image_dict:
                image_dict[curr_frame_hash] = data_contents
            timing_dict['timestamp'].append(time.time()-sweep_stim.start_time)
            timing_dict['xxhash'].append(curr_frame_hash)

            print time.time() - t0

        stimulus.update = new_update

    old_finalize = sweep_stim._finalize

    
    def new_finalize(*args, **kwargs):
        hash_filename_dict = collections.defaultdict(list)
        for key, val in image_dict.items():
            curr_array = np.array(m.cls_image(val, monitor))
            curr_frame_md5 = hashlib.md5(curr_array).hexdigest()
            save_file_name = os.path.abspath(os.path.join(output_dir, '%s.npy' % curr_frame_md5))
            np.save(save_file_name, curr_array)

            hash_filename_dict['file_name'].append(save_file_name)
            curr_frame_xxhash = xxhash.xxh64(val).digest()
            hash_filename_dict['xxhash'].append(curr_frame_xxhash)

        
        timing_df = pd.DataFrame(timing_dict)
        timing_df = pd.merge(timing_df, pd.DataFrame(hash_filename_dict), on='xxhash').drop('xxhash', axis=1)

        timing_df.to_csv(os.path.join(output_dir, 'stimtable.csv'))
        old_finalize(*args, **kwargs)
    sweep_stim._finalize = new_finalize