#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from visual_behavior.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df

if __name__ == '__main__':

    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis'
    manifest = pd.read_excel(os.path.join(cache_dir, 'opc_manifest.xlsx'))
    experiment_ids = manifest['Experiment ID(s)'].values

    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_id'])
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'trial_type'], use_events=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'])
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'], use_events=True)


    get_multi_session_mean_df(experiment_ids, cache_dir,
                                    conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                                      conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True, use_events=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                                      conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                                      conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True, use_events=True)





