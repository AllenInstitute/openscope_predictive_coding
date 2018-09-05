import os
import numpy as np
import hashlib
import collections
import pandas as pd
import time
import xxhash
from mss import mss
import ctypes
import platform

def add_frame_dir_linux(sweep_stim, output_dir='.'):
    
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




def add_frame_dir_windows(sweep_stim, output_dir='.'):
    
    PLAINMASK = 0x00ffffff
    CAPTUREBLT = 0x40000000
    DIB_RGB_COLORS = 0
    SRCCOPY = 0x00CC0020
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

def add_frame_dir(*args, **kwargs):

    if platform.system() == 'Windows':
        return add_frame_dir_windows(*args, **kwargs)
    elif platform.system() == 'Linux':
        return add_frame_dir_linux(*args, **kwargs)
    else:
        raise RuntimeError