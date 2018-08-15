import numpy as np
import os
import hashlib

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as si
import openscope_predictive_coding as opc
from openscope_predictive_coding.utilities import get_hash

def get_brain_observatory_templates(manifest_file = os.path.join(opc.boc_path, 'boc/manifest.json')):
    
    boc = BrainObservatoryCache(manifest_file=manifest_file)

    return_dict = {}

    # Natural movie one:
    nm1_path = os.path.join(opc.data_path, 'templates', 'NATURAL_MOVIE_ONE.npy')
    if os.path.exists(nm1_path):
        return_dict[si.NATURAL_MOVIE_ONE] = np.load(nm1_path)
    else:
        oeid = 511458874
        data_set = boc.get_ophys_experiment_data(oeid)
        return_dict[si.NATURAL_MOVIE_ONE] = data_set.get_stimulus_template(si.NATURAL_MOVIE_ONE)


    # Natural movie one:
    ns_path = os.path.join(opc.data_path, 'templates', 'NATURAL_SCENES.npy')
    if os.path.exists(ns_path):
        return_dict[si.NATURAL_SCENES] = np.load(ns_path)
    else:
        oeid = 511458874
        data_set = boc.get_ophys_experiment_data(oeid)
        return_dict[si.NATURAL_SCENES] = data_set.get_stimulus_template(si.NATURAL_SCENES)
    
    
    # Natural movie two:
    nm2_path = os.path.join(opc.data_path, 'templates', 'NATURAL_MOVIE_TWO.npy')
    if os.path.exists(nm2_path):
        return_dict[si.NATURAL_MOVIE_TWO] = np.load(nm2_path)
    else:
        oeid = 527550473
        data_set = boc.get_ophys_experiment_data(oeid)
        return_dict[si.NATURAL_MOVIE_TWO] = data_set.get_stimulus_template(si.NATURAL_MOVIE_TWO)

    hash_dict = {
                "natural_movie_two": "68e5976a140fe8400c6b7fe59073fe72",
                "natural_movie_one": "b174ad09736c870c6915baf82cf2c9ad",
                "natural_scenes": "b9a9a5284200f80b56ba6f4eecd34712"
                }

    assert len(return_dict) == 3

    for key, val in return_dict.items():
        assert hash_dict[key] == get_hash(val)


    return return_dict
    


