#!/usr/bin/env python
import os
import matplotlib
import pandas as pd
matplotlib.use('Agg')

import pandas as pd
from openscope_predictive_coding.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df

if __name__ == '__main__':

    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis'
    manifest_file = r"/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis/opc_production_manifest.xlsx"
    data = pd.read_excel(manifest_file)
    data = data[data['experiment_state'] == 'passed']
    experiment_ids = data.experiment_id.unique()
    experiment_ids = [int(expt_id) for expt_id in experiment_ids]

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='oddball',
                              conditions=['cell_specimen_id', 'image_id', 'oddball'], use_events=True)

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='occlusion',
                              conditions=['cell_specimen_id', 'image_id', 'fraction_occlusion'], use_events=True)

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='transition_control',
                              conditions=['cell_specimen_id', 'image_id', 'second_in_sequence'], use_events=True)

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='randomized_control_pre',
                              conditions=['cell_specimen_id', 'image_id', 'oddball'], use_events=True)

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='randomized_control_pre',
                              conditions=['cell_specimen_id', 'image_id', 'oddball'], use_events=True)

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='natural_movie_one',
                              conditions=['cell_specimen_id'], use_events=True)




