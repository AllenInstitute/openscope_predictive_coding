import numpy as np
import os
import hashlib

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as si
import openscope_predictive_coding as opc

def get_brain_observatory_templates(manifest_file = os.path.join(opc.data_path, 'boc/manifest.json')):
    
    boc = BrainObservatoryCache(manifest_file=manifest_file)

    return_dict = {}

    # oeid = boc.get_ophys_experiments(session_types=[si.THREE_SESSION_B])[0]['id']
    oeid = 511458874
    data_set = boc.get_ophys_experiment_data(oeid)
    return_dict[si.NATURAL_SCENES] = data_set.get_stimulus_template(si.NATURAL_SCENES)
    return_dict[si.NATURAL_MOVIE_ONE] = data_set.get_stimulus_template(si.NATURAL_MOVIE_ONE)

    # oeid = boc.get_ophys_experiments(session_types=[si.THREE_SESSION_C])[0]['id']
    oeid = 527550473
    data_set = boc.get_ophys_experiment_data(oeid)
    return_dict[si.NATURAL_MOVIE_TWO] = data_set.get_stimulus_template(si.NATURAL_MOVIE_TWO)

    return return_dict
    









# sys.exit()
# data_set = boc.get_ophys_experiment_data(oeid)


# oeid = boc.get_ophys_experiments(session_types=[si.THREE_SESSION_C])[0]['id']
# data_set = boc.get_ophys_experiment_data(oeid)
# save_file_name = os.path.join(save_dir, 'NATURAL_MOVIE_TWO.npy')
# movie_data = data_set.get_stimulus_template(si.NATURAL_MOVIE_TWO)
# print hashlib.md5(movie_data).hexdigest()
# np.save(save_file_name, movie_data)
