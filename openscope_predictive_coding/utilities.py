import random
import os
import numpy as np
import hashlib
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
import tempfile
import scipy.optimize as sopt
import collections
import functools
import json
import skimage.io as sio
import pandas as pd
import sys

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
    plt.close()
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
    
    src_sequence = list(src_sequence)

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

def tiff_to_numpy(input_file_name):
    assert input_file_name[-4:] == '.tif'
    save_file_name = input_file_name[:-4] + '.npy'
    print save_file_name
    np.save(save_file_name, sio.imread(input_file_name))

def file_name_to_stimulus_hash(file_name):
    f, file_extension = os.path.splitext(os.path.basename(file_name))
    return f.split('_')[-1]

def get_interval_table_v1(session, data_path=opc.stimtable_path):

    occlusion_metadata_df = opc.stimulus.get_occlusion_metadata()
    _, pilot_randomized_control_full_sequence = opc.stimulus.get_pilot_randomized_control(session, data_path=data_path)

    hash_dict_rev = {hash:key for key, hash in opc.stimulus.hash_dict.items()}
    data_block_dict = {}
    
    file_path = os.path.join(data_path, 'interval_data_{session}.json'.format(session=session))

    data = json.load(open(file_path, 'r'))

    df_dict = collections.defaultdict(list)
    for session_block_name in data.keys():
        for interval_data in data[session_block_name]:
            (t_start, t_end), file_name, frame_length, number_of_runs = interval_data
            curr_hash = file_name_to_stimulus_hash(file_name)

            if curr_hash not in data_block_dict:
                data_block_dict[curr_hash] = np.load(file_name)

            stimulus_key = hash_dict_rev[curr_hash]
            if isinstance(stimulus_key, (tuple,)):
                image_id_tuple = stimulus_key
                data_file_index_tuple = range(len(image_id_tuple))
            else:
                image_id_tuple = None
                data_file_index_tuple = tuple(range(data_block_dict[curr_hash].shape[0]))

                if t_end - t_start != frame_length*len(data_file_index_tuple):

                    # Occlusion
                    data_file_index_tuple = stimulus_key

            # Build Dataframe:
            df_dict['stimulus_key'].append(stimulus_key)
            df_dict['start'].append(t_start)
            df_dict['end'].append(t_end)
            df_dict['frame_length'].append(frame_length)
            df_dict['image_id_tuple'].append(image_id_tuple)
            df_dict['data_file_index_tuple'].append(data_file_index_tuple)
            df_dict['data_file_name'].append(file_name)
            df_dict['session_block_name'].append(session_block_name)
            
            

    df_main = df = pd.DataFrame(df_dict)
    expanded_df_list = []

    # Expand occlusion:
    sorted_occlusion_df = df[df['data_file_index_tuple']=='ophys_pilot_occlusion'].sort_values('start').reset_index().rename(columns={'index':'fk'})[['fk']]
    occlusion_df = sorted_occlusion_df.join(occlusion_metadata_df).set_index('fk').join(df).drop(['data_file_index_tuple','data_file_name', 'image_id_tuple', 'stimulus_key', 'session_block_name'], axis=1)
    occlusion_df = occlusion_df.rename(columns={'start':'start_time', 'end':'end_time', 'frame_length':'duration'})
    occlusion_df['data_file_index'] = range(len(occlusion_df))
    expanded_df_list.append(occlusion_df)
    df = df[~df.index.isin(set.union(*[set(x.index) for x in expanded_df_list]))]
    
    # Expand Natural Movies:
    for movie_stimulus_key in [x for x in df['stimulus_key'].unique() if 'natural_movie' in x]:

        new_df_dict = collections.defaultdict(list)
        sorted_movie_df = df[df['stimulus_key']==movie_stimulus_key].sort_values('start').reset_index().rename(columns={'index':'fk'})
        for repeat, row in sorted_movie_df.iterrows():
            assert row['end'] - row['start'] == row['frame_length']*len(row['data_file_index_tuple'])
    
            for data_file_index in row['data_file_index_tuple']:
                start_time = row['start']+data_file_index*row['frame_length']
                end_time = start_time + row['frame_length']
                
                new_df_dict['duration'].append(end_time - start_time)
                new_df_dict['start_time'].append(start_time)
                new_df_dict['end_time'].append(end_time)
                new_df_dict['repeat'].append(repeat)
                new_df_dict['data_file_index'].append(data_file_index)
                new_df_dict['fk'].append(row['fk'])

        natural_movie_df = pd.DataFrame(new_df_dict).set_index('fk')
        expanded_df_list.append(natural_movie_df)
        df = df[~df.index.isin(set.union(*[set(x.index) for x in expanded_df_list]))]

    # Expand randomized control:
    for randomized_control_stimulus_key in [x for x in df['stimulus_key'].unique() if 'ophys_pilot_randomized_control' in x]:

        new_df_dict = collections.defaultdict(list)
        randomized_control_df = df[df['stimulus_key']==randomized_control_stimulus_key].sort_values('start').reset_index().rename(columns={'index':'fk'})
        for _, row in randomized_control_df.iterrows():
            assert row['end'] - row['start'] == row['frame_length']*len(row['data_file_index_tuple'])

            for data_file_index in row['data_file_index_tuple']:
                start_time = row['start']+data_file_index*row['frame_length']
                end_time = start_time + row['frame_length']
                
                new_df_dict['duration'].append(end_time - start_time)
                new_df_dict['image_id'].append(pilot_randomized_control_full_sequence[data_file_index])
                new_df_dict['start_time'].append(start_time)
                new_df_dict['end_time'].append(end_time)
                new_df_dict['data_file_index'].append(data_file_index)
                new_df_dict['fk'].append(row['fk'])


        randomized_control = pd.DataFrame(new_df_dict).set_index('fk')
        expanded_df_list.append(randomized_control)
        df = df[~df.index.isin(set.union(*[set(x.index) for x in expanded_df_list]))]

    # Expand tuples:
    new_df_dict = collections.defaultdict(list)
    for fk, row in df.iterrows():
        assert row['end'] - row['start'] == row['frame_length']*len(row['data_file_index_tuple'])
        assert isinstance(row['stimulus_key'], tuple)

        for data_file_index in row['data_file_index_tuple']:
            start_time = row['start']+data_file_index*row['frame_length']
            end_time = start_time + row['frame_length']
            
            image_id = row['image_id_tuple'][data_file_index]
            new_df_dict['duration'].append(end_time - start_time)
            new_df_dict['start_time'].append(start_time)
            new_df_dict['end_time'].append(end_time)
            new_df_dict['data_file_index'].append(data_file_index)
            new_df_dict['image_id'].append(image_id)
            new_df_dict['fk'].append(fk)

    individual_frame_df = pd.DataFrame(new_df_dict).set_index('fk')
    expanded_df_list.append(individual_frame_df)
    df = df[~df.index.isin(set.union(*[set(x.index) for x in expanded_df_list]))]
    assert len(df) == 0

    b = df_main.drop(['data_file_index_tuple','image_id_tuple', 'start', 'end', 'frame_length'], axis=1)
    b.index.name = 'fk'

    expanded_df_list_join = [a.join(b) for a in expanded_df_list]
    
    df_final = pd.concat(expanded_df_list_join).sort_values('start_time')
    df_final['data_file_index'] = df_final['data_file_index'].astype(np.int)
    df_final['session_type'] = session
    return df_final

def running_group(running_keys, indices_to_group):
    assert len(running_keys) == len(indices_to_group)

    old=None
    key_list = []
    tmp = []
    for key, other in zip(running_keys, indices_to_group):
        if key == old:
            tmp[-1].append(other)
        else:
            tmp.append([other])
            key_list.append(key)
        old = key

    return key_list, tmp

def get_interval_table(version=1, **kwargs):
    
    if version==1:
        session = kwargs['session']
        data_path = kwargs.get('data_path', opc.stimtable_path)
        return get_interval_table_v1(session, data_path=data_path)
    else:
        raise RuntimeError


def pickle_file_to_interval_table(pickle_file_name, version=1):

    if 'StimA' in pickle_file_name:
        stimtable_df = get_interval_table(version=version, session='A')
    elif 'StimB' in pickle_file_name:
        stimtable_df = get_interval_table(version=version, session='B')
    elif 'StimC' in pickle_file_name:
        stimtable_df = get_interval_table(version=version, session='C')
    else:
        raise

    data = pickle.load(open(pickle_file_name, 'r'))
    
    df_list = []
    for ii, stimuli in enumerate(data['stimuli']):

        data_file_name = stimuli['movie_path'].replace('\\', '/')
        curr_stimtable = stimtable_df[stimtable_df['data_file_name']==data_file_name].reset_index().rename(columns={'index':'lk'})
        frame_inds = np.where(stimuli['frame_list'] != -1)
        data_file_indices = stimuli['frame_list'][frame_inds]

        data_dict = collections.defaultdict(list)
        data_file_indices_flattened, running_frame_ind_list = running_group(data_file_indices, one(frame_inds))
        for curr_file_indices, curr_running_frames in zip(data_file_indices_flattened, running_frame_ind_list):
    
            data_dict['frame_list'].append(curr_running_frames)
        
        run_grouped_df = pd.DataFrame(data_dict)

        assert len(run_grouped_df) == len(curr_stimtable)
        curr_file_df = curr_stimtable.join(run_grouped_df).set_index('lk')
        df_list.append(curr_file_df)

    df_final = pd.concat(df_list).sort_values(['start_time', 'end_time']).drop(['start_time', 'end_time'], axis=1)

    df_final['start_frame'] = df_final['frame_list'].map(lambda x: x[0])
    df_final['end_frame_inclusive'] = df_final['frame_list'].map(lambda x: x[-1])

    return df_final


if __name__ == "__main__":
    
    # Debugging:

    f = '/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis/746271249_400524_180906_RSP_75_Slc17a7_2P1_20180906_400524_StimB/746004188_400524_20180906_stim.pkl'
    df = pickle_file_to_interval_table(f)