from openscope_predictive_coding.ophys.dataset.openscope_predictive_coding_dataset import OpenScopePredictiveCodingDataset
from openscope_predictive_coding.ophys.response_analysis.response_analysis import ResponseAnalysis

import matplotlib
import logging

matplotlib.use('Agg')


def create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True):
    print('saving ', str(experiment_id), 'to', cache_dir)
    dataset = OpenScopePredictiveCodingDataset(experiment_id, cache_dir)
    analysis = ResponseAnalysis(dataset, overwrite_analysis_files)




if __name__ == '__main__':
    # import sys
    #
    # experiment_id = sys.argv[1]
    # cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis'
    # # cache_dir = r'/allen/aibs/informatics/swdb2018/visual_behavior'
    # create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True)

    experiment_id = 746271249
    # experiment_id = 746270939
    # experiment_id = 746271665

    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\opc\opc_analysis'
    create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True)
