import os
import numpy as np
import hashlib
import collections
import pandas as pd
import time
import xxhash
from mss import mss
import ctypes






PLAINMASK = 0x00ffffff
CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0
SRCCOPY = 0x00CC0020
ZPIXMAP = 2
width, height, top, left = 1920, 1200, 0, 0
monitor = {'width': width, 'top': top, 'height': height, 'left': left}


def add_frame_dir(sweep_stim, output_dir='.'):
    
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

            gdi = ctypes.windll.gdi32
            width, height = monitor["width"], monitor["height"]

            if (m._bbox["height"], m._bbox["width"]) != (height, width):
                m._bbox = monitor
                m._bmi.bmiHeader.biWidth = width
                m._bmi.bmiHeader.biHeight = -height

            m._srcdc = ctypes.windll.user32.GetWindowDC(0)
            m._memdc = ctypes.windll.gdi32.CreateCompatibleDC(m._srcdc)
            m._data = ctypes.create_string_buffer(width * height * 4)  # [2]
            m._bmp = gdi.CreateCompatibleBitmap(m._srcdc, width, height)
            gdi.SelectObject(m._memdc, m._bmp)

            gdi.BitBlt(
                m._memdc,
                0,
                0,
                width,
                height,
                m._srcdc,
                monitor["left"],
                monitor["top"],
                SRCCOPY | CAPTUREBLT,
            )
            bits = gdi.GetDIBits(
                m._memdc, m._bmp, 0, height, m._data, m._bmi, DIB_RGB_COLORS
            )
            data_contents = m._data
            if bits != height:
                raise ScreenShotError("gdi32.GetDIBits() failed.")

            curr_frame_hash = xxhash.xxh64(data_contents).digest()

            for attr in (m._memdc, m._srcdc, m._data, m._bmp):
                if attr:
                    ctypes.windll.gdi32.DeleteObject(attr)

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
            # curr_frame_xxhash = xxhash.xxh64(curr_frame_downsample).digest()
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
