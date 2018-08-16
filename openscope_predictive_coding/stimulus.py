import numpy as np
import os
import hashlib

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as si
import openscope_predictive_coding as opc
from openscope_predictive_coding.utilities import get_hash, apply_warp_natural_scene, apply_warp_natural_movie, luminance_match, downsample_monitor_to_template, get_shuffled_repeated_sequence, generate_sequence_block, seq_to_str
from utilities import memoized


default_manifest = os.path.join(opc.boc_path, 'boc/manifest.json')
default_data_path = opc.data_path


TEMPLATE_LIST = [si.NATURAL_MOVIE_ONE, si.NATURAL_MOVIE_TWO, si.NATURAL_SCENES]
TEMPLATE_LIST_WARPED = ["%s_warped" % s for s in TEMPLATE_LIST]
NATURAL_SCENES_WARPED = si.NATURAL_SCENES + '_warped'
HABITUATION_PILOT_RANDOMIZED_ODDBALL = 'habituation_pilot_randomized_oddball'
HABITUATION_PILOT_DATA = [HABITUATION_PILOT_RANDOMIZED_ODDBALL]

stimulus_oeid_dict = {
                     si.NATURAL_MOVIE_ONE: 511458874,
                     si.NATURAL_SCENES: 511458874,
                     si.NATURAL_MOVIE_TWO: 527550473
                     }

hash_dict = {
            si.NATURAL_MOVIE_ONE: "b174ad09736c870c6915baf82cf2c9ad",
            si.NATURAL_MOVIE_TWO: "68e5976a140fe8400c6b7fe59073fe72",
            si.NATURAL_SCENES: "b9a9a5284200f80b56ba6f4eecd34712",
            si.NATURAL_MOVIE_TWO + '_warped': '92f1fe36e2c118761cbebcebcc6cd076',
            si.NATURAL_MOVIE_ONE + '_warped': '77ee4ecd0dc856c80cf24621303dd080',
            NATURAL_SCENES_WARPED: '8ba4262b06ec81c3ec8d3d7d7831e564',
            HABITUATION_PILOT_RANDOMIZED_ODDBALL: '5cd9854e9cb07a427180d6e130c148ab',
            opc.SEQUENCE_IMAGES['A']: '2950e8d1e5187ce65ac40f5381be0b3f'
            }


STIMULUS_LIST = TEMPLATE_LIST+TEMPLATE_LIST_WARPED+HABITUATION_PILOT_DATA

assert set(stimulus_oeid_dict.keys()) == set(TEMPLATE_LIST)
assert set(hash_dict.keys()) == set(STIMULUS_LIST+[opc.SEQUENCE_IMAGES['A']])
assert NATURAL_SCENES_WARPED in TEMPLATE_LIST_WARPED

def get_stimulus_path(stimulus_key, data_path=default_data_path, append_hash=True):
    if isinstance(stimulus_key, tuple):
        stimulus = stimulus_key
        stimulus_key = seq_to_str(stimulus_key)
    else:
        stimulus = stimulus_key

    if append_hash:
        return os.path.join(data_path, '%s_%s.npy' % (stimulus_key, hash_dict[stimulus]))
    else:
        return os.path.join(data_path, '%s.npy' % (stimulus_key, ))

def remove_warped_from_stimulus_key(stimulus_name):
    assert stimulus_name in STIMULUS_LIST

    split_stimulus = stimulus_name.split('_')
    assert split_stimulus[-1] == 'warped'
    return '_'.join(split_stimulus[:-1])
    
@memoized
def get_stimulus_template_brain_observatory(stimulus, data_path=default_data_path, manifest_file=default_manifest):
    
    boc = BrainObservatoryCache(manifest_file=manifest_file)
    dataset_path = get_stimulus_path(stimulus, data_path=data_path)

    if os.path.exists(dataset_path):
        data = np.load(dataset_path)
    else:
        oeid = stimulus_oeid_dict[stimulus]
        data_set = boc.get_ophys_experiment_data(oeid)
        data = data_set.get_stimulus_template(stimulus)
        assert hash_dict[stimulus] == get_hash(data)
        np.save(dataset_path, data)

    assert hash_dict[stimulus] == get_hash(data)
    return data
    
@memoized
def get_stimulus_template_warped(stimulus_key, data_path=default_data_path, manifest_file=default_manifest):
    
    stimulus_key_prewarp = remove_warped_from_stimulus_key(stimulus_key)

    data = get_stimulus_template_brain_observatory(stimulus_key_prewarp, data_path, manifest_file)
    data_warp_path = get_stimulus_path(stimulus_key, data_path=data_path)
    if os.path.exists(data_warp_path):
        data_warp = np.load(data_warp_path)
    else:
        data_warp = np.empty((data.shape[0], opc.IMAGE_H, opc.IMAGE_W), dtype=np.uint8)
        for fi, img in enumerate(data):
            if stimulus_key_prewarp in si.NATURAL_MOVIE_STIMULUS_TYPES:
                img_warp = apply_warp_natural_movie(img)
                img_warp_ds = downsample_monitor_to_template(img_warp)
                data_warp[fi,:,:] = img_warp_ds
            elif stimulus_key_prewarp == si.NATURAL_SCENES:
                img_warp = apply_warp_natural_scene(img)
                img_warp_lm = luminance_match(img_warp)
                img_warp_lm_ds = downsample_monitor_to_template(img_warp_lm)[::-1,:]
                data_warp[fi,:,:] = img_warp_lm_ds
            else:
                raise RuntimeError
            print stimulus_key, fi
        assert hash_dict[stimulus_key] == get_hash(data_warp)
        np.save(data_warp_path, data_warp)
    
    assert hash_dict[stimulus_key] == get_hash(data_warp)
    return data_warp

@memoized
def get_stimulus_randomized_oddball(**kwargs):
    data_path = kwargs.get('data_path', default_data_path)

    dataset_path = get_stimulus_path(HABITUATION_PILOT_RANDOMIZED_ODDBALL, data_path=data_path)

    if os.path.exists(dataset_path):
        data = np.load(dataset_path)
    else:
        habituation_oddball_full_sequence = get_shuffled_repeated_sequence(opc.HABITUATED_ODDBALL_IMAGES, 100, seed=0)
        src_image_data = get_stimulus_template(NATURAL_SCENES_WARPED)
        data = generate_sequence_block(habituation_oddball_full_sequence, src_image_data)
        assert hash_dict[HABITUATION_PILOT_RANDOMIZED_ODDBALL] == get_hash(data)
        np.save(dataset_path, data)

    assert hash_dict[HABITUATION_PILOT_RANDOMIZED_ODDBALL] == get_hash(data)
    return data

@memoized
def get_sequence_template(sequence, **kwargs):
    data_path = kwargs.get('data_path', default_data_path)
    
    dataset_path = get_stimulus_path(sequence, data_path=data_path)

    if os.path.exists(dataset_path):
        data = np.load(dataset_path)
    else:
    
        src_image_data = get_stimulus_template(NATURAL_SCENES_WARPED)
        data = generate_sequence_block(sequence, src_image_data)
        assert hash_dict[sequence] == get_hash(data)
        np.save(dataset_path, data)
    
    assert hash_dict[sequence] == get_hash(data)
    return data


@memoized
def get_stimulus_template(stimulus, **kwargs):
    
    if stimulus in TEMPLATE_LIST:
        data = get_stimulus_template_brain_observatory(stimulus, **kwargs)
        assert hash_dict[stimulus] == get_hash(data)
    elif stimulus in TEMPLATE_LIST_WARPED:
        data = get_stimulus_template_warped(stimulus, **kwargs)
        assert hash_dict[stimulus] == get_hash(data)
    elif stimulus == HABITUATION_PILOT_RANDOMIZED_ODDBALL:
        data = get_stimulus_randomized_oddball(**kwargs)
        assert hash_dict[stimulus] == get_hash(data)
    elif isinstance(stimulus, tuple):
        sequence = stimulus
        data = get_sequence_template(sequence, **kwargs)
    else:
        raise RuntimeError
            
    return data


if __name__ == "__main__":
    
    # for stimulus in STIMULUS_LIST:
    # template = get_stimulus_template(si.NATURAL_SCENES+'_warped')
    template = get_stimulus_template(opc.HABITUATED_SEQUENCE_IMAGES)

    # get_stimulus_template(si.NATURAL_SCENES + '_warped')


