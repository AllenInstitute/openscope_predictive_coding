import random
import os
import numpy as np
import hashlib
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
import tempfile
import scipy.optimize as sopt
import collections
import functools

import openscope_predictive_coding as opc
import allensdk.brain_observatory.stimulus_info as si

monitor = si.BrainObservatoryMonitor()

def seq_to_str(sequence):
    return '_'.join([str(ii) for ii in sequence])


def downsample_monitor_to_template(img):
    
    if len(img.shape) == 2:
        assert img.shape == (opc.SCREEN_H, opc.SCREEN_W)
        new_data = img[::2, ::2]
        assert new_data.shape == (opc.IMAGE_H, opc.IMAGE_W)
        return new_data
    elif len(img.shape) == 3:
        raise NotImplementedError
    else:
        raise Exception


def one(x):
    assert len(x) == 1
    if isinstance(x,set):
        return list(x)[0]
    else:
        return x[0]

def apply_warp_on_monitor(img):

    fig, ax = plt.subplots(1,1)
    monitor.show_image(img, ax=ax, warp=True, mask=False, show=False)
    img_warped = one([obj for obj in ax.get_children() if isinstance(obj, mpl.image.AxesImage)]).get_array()
    assert img_warped.shape == si.MONITOR_DIMENSIONS
    return img_warped

def apply_warp_natural_scene(img):
    assert img.shape == si.NATURAL_SCENES_PIXELS
    img_screen = monitor.natural_scene_image_to_screen(img)
    return apply_warp_on_monitor(img_screen)

def apply_warp_natural_movie(img):
    assert img.shape == si.NATURAL_MOVIE_DIMENSIONS
    img_screen = monitor.natural_movie_image_to_screen(img, origin='upper')
    return apply_warp_on_monitor(img_screen)

def get_hash(data):

    return hashlib.md5(data).hexdigest()

def get_shuffled_repeated_sequence(src_sequence, number_of_shuffles, seed=None):
    
    if seed is not None:
        random.seed(seed)
    
    new_sequence = []
    for x in range(number_of_shuffles):
        random.shuffle(src_sequence)
        new_sequence += list(src_sequence)

    return new_sequence

def generate_oddball_block_timing_dict(base_seq, oddball_list, num_cycles_per_repeat=20, oddball_cycle_min=10, oddball_cycle_max=19, num_repeats_per_oddball=10, frame_length=.25, expected_duration=None, seed=None):
    
    # Force to be zero, to avoid confusion when making script
    t0=0

    if seed is not None:
        random.seed(seed)

    if not expected_duration is None:
        np.testing.assert_approx_equal(expected_duration, len(base_seq)*len(oddball_list)*num_cycles_per_repeat*num_repeats_per_oddball*frame_length)
    
    delta_t = frame_length*len(base_seq)
    t = t0
    timing_dict = defaultdict(list)
    for _ in range(num_repeats_per_oddball):
        
        oddball_list_shuffle = [x for x in oddball_list]
        random.shuffle(oddball_list_shuffle)

        for oddball in oddball_list_shuffle:
            
            cycle_to_replace = random.choice(range(oddball_cycle_min-1,oddball_cycle_max)) 
            curr_repeat = [[x for x in base_seq] for _ in range(num_cycles_per_repeat)]
            curr_repeat[cycle_to_replace][-1] = oddball

            for curr_cycle in curr_repeat:
                timing_dict[tuple(curr_cycle)].append((t, t+delta_t))
                t += delta_t            


    # Lots of Double-checking:
    number_of_total_cycles = 0
    timing_hwm = -float('inf')
    for pattern, timing_list in timing_dict.items():
        number_of_total_cycles += len(timing_list)

        if pattern != tuple(base_seq):
            assert len(timing_list) == num_repeats_per_oddball
        for timing_tuple in timing_list:
            assert timing_tuple[0] < timing_tuple[1]
            np.testing.assert_approx_equal( (timing_tuple[1] - timing_tuple[0])/len(pattern), frame_length)
            if timing_tuple[1] > timing_hwm:
                timing_hwm = timing_tuple[1]
    assert number_of_total_cycles == len(oddball_list)*num_cycles_per_repeat*num_repeats_per_oddball
    assert len(timing_dict[tuple(base_seq)]) == number_of_total_cycles - num_repeats_per_oddball*len(oddball_list)

    np.testing.assert_approx_equal(t0+len(base_seq)*len(oddball_list)*num_cycles_per_repeat*num_repeats_per_oddball*frame_length, timing_hwm)

    return timing_dict

def generate_sequence_block(base_seq, src_image_data):

    N = len(base_seq)
    h, w = src_image_data.shape[1:]
    data_block = np.zeros((N,h,w), dtype=np.uint8)
    for ii, idx in enumerate(base_seq):
        data_block[ii,:,:] = src_image_data[idx,:,:]

    return data_block

def generate_pair_block_timing_dict(pair_list, num_repeats=30, frame_length=.25, expected_duration=None, seed=None):

    if seed is not None:
        random.seed(seed)

    for x in pair_list:
        assert len(x) == 2

    timing_list = []
    t = 0.
    delta_t = t + frame_length*2
    for ii in range(len(pair_list)*num_repeats):
        timing_list.append((t, t+delta_t))
        t += delta_t
    
    timing_dict = defaultdict(list)
    for ii in range(num_repeats):
        curr_repeat_time_list = timing_list[ii*len(pair_list):(ii+1)*len(pair_list)]
        tmp_pair_list = [x for x in pair_list]
        random.shuffle(tmp_pair_list)
        for curr_pair, curr_repeat_time in zip(tmp_pair_list, curr_repeat_time_list):
            timing_dict[curr_pair].append(curr_repeat_time)
    
    for curr_timing_list in timing_dict.values():
        curr_timing_list.sort(key=lambda x:x[0])
    
    return timing_dict

def linear_transform_image(img, m_M):
    
    m = float(m_M[0])
    M = 255.-float(m_M[1])
    
    return (img-img.min())*(M-m)/(img.max()-img.min()) + m

def luminance_match(img):
    
    def f(m_M):
        
        tmp = linear_transform_image(img, m_M)
        aa = tmp.mean()
        bb = tmp.std()/tmp.mean()

        return 100*((tmp.mean()-127.)/aa)**2 + ((tmp.std()/tmp.mean()-.6)/bb)**2

    x_res = sopt.minimize(f, (0,0), bounds=((0,255),(0,255),)).x
    return linear_transform_image(img, x_res)

def run_camstim_debug(img_stack, timing_list, frame_length, runs):
    
    assert (img_stack.dtype) == 'uint8'
    assert len(img_stack.shape) == 3
    
    from camstim import SweepStim, MovieStim
    from camstim import Foraging
    from camstim import Window

    # Create display window, warped
    window = Window(fullscr=True,
                    monitor='Gamma1.Luminance50',
                    screen=0
                    )

    tmp_dir = tempfile.mkdtemp()

    timing_list = sorted(timing_list, key=lambda x:x[0])
    movie_path = '%s.npy' % get_hash(img_stack)
    np.save(movie_path, img_stack)
    movie =  MovieStim(movie_path=movie_path,
                        window=window,
                        frame_length=frame_length,
                        size=(opc.SCREEN_W, opc.SCREEN_H),
                        start_time=0.0,
                        stop_time=None,
                        flip_v=True,
                        runs=runs)

    movie.set_display_sequence(timing_list)
    stimuli = [movie]

    ss = SweepStim(window,
                stimuli=stimuli,
                pre_blank_sec=0,
                post_blank_sec=0)

    f = Foraging(window=window,
                auto_update=False,
                nidaq_tasks={'digital_input': ss.di,
                            'digital_output': ss.do,})  #share di and do with SS
    ss.add_item(f, "foraging")
    ss.run()




class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).

   https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)