from openscope_predictive_coding.ophys.dataset.openscope_predictive_coding_dataset import OpenScopePredictiveCodingDataset
from openscope_predictive_coding.ophys.response_analysis.response_analysis import ResponseAnalysis
import openscope_predictive_coding.ophys.response_analysis.utilities as ut
import pandas as pd
import logging
import os


# logger = logging.getLogger(__name__)


def get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='oddball',
                              conditions=['cell_specimen_id', 'image_id', 'oddball'],
                              flashes=False, use_events=True):
    mega_mdf = pd.DataFrame()
    for experiment_id in experiment_ids:
        print(experiment_id)
        dataset = OpenScopePredictiveCodingDataset(experiment_id, cache_dir=cache_dir)
        analysis = ResponseAnalysis(dataset, preload_response_dfs=False, overwrite_analysis_files=False, use_events=use_events)
        try:
            # df = analysis.response_df_dict[session_block_name]
            df = analysis.get_response_df(session_block_name)
            if session_block_name is 'transition_control':
                df['second_in_sequence'] = [True if df.iloc[row].stimulus_key[1] == df.iloc[row].image_id else False
                                             for row in range(0, len(df))]
            mdf = ut.get_mean_df(df, conditions=conditions)
            mdf = ut.add_retrogradely_labeled_column_to_df(mdf, cache_dir)
            mdf = ut.add_projection_pathway_to_df(mdf, cache_dir)
            mdf['experiment_id'] = dataset.experiment_id
            mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
            mega_mdf = pd.concat([mega_mdf, mdf])
        except:
            print('problem for',experiment_id)
    if use_events:
        suffix = '_events'
    else:
        suffix = ''
    if 'level_0' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='level_0')
    if 'index' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='index')

    mega_mdf.to_hdf(os.path.join(cache_dir, 'multi_session_summary_dfs',
                                 'mean_'+session_block_name+'_df'+suffix+'.h5'), key='df')



if __name__ == '__main__':
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\opc\opc_analysis'
    manifest_file = r"\\allen\programs\braintv\workgroups\nc-ophys\opc\opc_analysis\opc_production_manifest.xlsx"
    data = pd.read_excel(manifest_file)
    data = data[data['experiment_state'] == 'passed']
    experiment_ids = data.experiment_id.unique()
    experiment_ids = np.asarray([int(expt_id) for expt_id in experiment_ids])

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name = 'oddball',
                              conditions=['cell_specimen_id', 'image_id', 'oddball'])

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='occlusion',
                              conditions=['cell_specimen_id', 'image_id', 'fraction_occlusion'])

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='transition_control',
                              conditions=['cell_specimen_id', 'image_id', 'second_in_sequence'])

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='randomized_control_pre',
                              conditions=['cell_specimen_id', 'image_id', 'oddball'])

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='randomized_control_pre',
                                  conditions=['cell_specimen_id', 'image_id', 'oddball'])

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='natural_movie_one',
                                  conditions=['cell_specimen_id'])

