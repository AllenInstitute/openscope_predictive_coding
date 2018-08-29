import json
import numpy as np
import warnings
import pandas as pd
import collections
import sys
import os

from openscope_predictive_coding.stimulus import hash_lookup, get_pilot_randomized_control, get_occlusion_metadata
import openscope_predictive_coding as opc
from openscope_predictive_coding.utilities import memoized
import allensdk.brain_observatory.stimulus_info as si

@memoized
def get_stimtable(session_type):

    data_path = opc.data_path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(dir_path, 'scripts','ophys','interval_data_%s.json' % session_type)
    interval_data = json.load(open(file_name, 'r'))
    _, pilot_randomized_control_full_sequence = get_pilot_randomized_control(session_type, data_path=data_path)
    occlusion_df = get_occlusion_metadata(data_path=data_path)

    file_name_template_dict = {}
    df_data = collections.defaultdict(list)
    for block_key, interval_list in interval_data.items():
        for interval, data_file_name, frame_length, runs in interval_list:

            stimulus_hash = data_file_name.split('_')[-1].split('.')[0]
            stimulus_key = hash_lookup(stimulus_hash)

            if data_file_name not in file_name_template_dict:
                file_name_template_dict[data_file_name] = np.load(data_file_name)

            for fi, subinterval_start in enumerate(np.arange(interval[0], interval[1], frame_length)):
                subinterval_end = subinterval_start + frame_length
                df_data['start'].append(subinterval_start)
                df_data['end'].append(subinterval_end)
                df_data['template_frame_index'].append(fi)
                df_data['template_file'].append(data_file_name)
                df_data['stimulus_key'].append(stimulus_key)
                df_data['stimulus_hash'].append(stimulus_hash)
                df_data['block_key'].append(block_key)

                if block_key in ['randomized_control_post', 'randomized_control_pre']:
                    df_data['scene_id'].append(pilot_randomized_control_full_sequence[fi])
                    df_data['fraction_occlusion'].append(0)
                elif isinstance(stimulus_key, tuple):
                    df_data['scene_id'].append(stimulus_key[fi])
                    df_data['fraction_occlusion'].append(0)
                elif block_key in [si.NATURAL_MOVIE_ONE, si.NATURAL_MOVIE_TWO]:
                    df_data['scene_id'].append(None)
                    df_data['fraction_occlusion'].append(None)
                elif block_key == 'occlusion':
                    tmp = occlusion_df.iloc[fi].to_dict()
                    df_data['scene_id'].append(int(tmp['image_index']))
                    df_data['fraction_occlusion'].append(int(tmp['fraction_occlusion']))

                else:
                    raise Exception

    return pd.DataFrame(df_data)
