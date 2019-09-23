from openscope_predictive_coding.ophys.dataset.openscope_predictive_coding_dataset import OpenScopePredictiveCodingDataset
from openscope_predictive_coding.ophys.response_analysis.response_analysis import ResponseAnalysis
import openscope_predictive_coding.ophys.response_analysis.utilities as ut
import pandas as pd
import logging
import os


# logger = logging.getLogger(__name__)


def get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='oddball',
                              conditions=['cell_specimen_id', 'image_id', 'oddball'],
                              flashes=False, use_events=False):
    mega_mdf = pd.DataFrame()
    for experiment_id in experiment_ids:
        print(experiment_id)
        dataset = OpenScopePredictiveCodingDataset(experiment_id, cache_dir=cache_dir)
        analysis = ResponseAnalysis(dataset, preload_response_dfs=False)
        try:
            # df = analysis.response_df_dict[session_block_name]
            df = analysis.get_response_df(session_block_name)
            if session_block_name is 'transition_control':
                df['second_in_sequence'] = [True if df.iloc[row].stimulus_key[1] == df.iloc[row].image_id else False
                                             for row in range(0, len(df))]
            mdf = ut.get_mean_df(df, conditions=conditions)
            mdf['experiment_id'] = dataset.experiment_id
            mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
            mega_mdf = pd.concat([mega_mdf, mdf])
        except:
            print('problem for',experiment_id)
    if 'level_0' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='level_0')
    if 'index' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='index')

    mega_mdf.to_hdf(os.path.join(cache_dir, 'multi_session_summary_dfs',
                                 'mean_'+session_block_name+'_df.h5'), key='df')



if __name__ == '__main__':
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\opc\opc_analysis'
    # manifest = pd.read_csv(os.path.join(cache_dir, 'visual_behavior_data_manifest.csv'))
    # experiment_ids = manifest.experiment_id.values
    experiment_ids = [768898762, 775058863, 775613721, 776727982, 813071318, 816795279,
                     817251851, 818894752, 826576489, 827232898, 828956958, 829411383,
                     848005700, 848690810, 848006710, 848691390, 830688102, 832601977,
                     832617299, 833599179, 830075254, 830688059, 831312165, 832107135,
                     833614835, 834260382, 838330377, 835642229, 835654507, 836246273,
                     836891984, 833626456, 836248932, 837630919, 837287590, 827235482,
                     828959377, 829417358, 831314921, 833612445, 835660148, 836253258,
                     836906598, 834244626, 836250018, 836895367, 837285285, 833611925,
                     834251985, 836890936, 837283374]

    # get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name = 'oddball',
    #                           conditions=['cell_specimen_id', 'image_id', 'oddball'])
    # # #
    # get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='occlusion',
    #                           conditions=['cell_specimen_id', 'image_id', 'fraction_occlusion'])

    get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='transition_control',
                              conditions=['cell_specimen_id', 'image_id', 'second_in_sequence'])
    #
    # get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='randomized_control_pre',
    #                           conditions=['cell_specimen_id', 'image_id', 'oddball'])

    # #
    # get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='randomized_control_pre',
    #                               conditions=['cell_specimen_id', 'image_id', 'oddball'])
    # #

    # get_multi_session_mean_df(experiment_ids, cache_dir, session_block_name='natural_movie_one',
    #                               conditions=['cell_specimen_id'])

