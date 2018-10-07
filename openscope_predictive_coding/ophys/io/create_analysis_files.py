from __future__ import print_function
from openscope_predictive_coding.ophys.dataset.openscope_predictive_coding_dataset import OpenScopePredictiveCodingDataset
from openscope_predictive_coding.ophys.response_analysis.response_analysis import ResponseAnalysis
import openscope_predictive_coding.ophys.plotting.summary_figures as sf
import openscope_predictive_coding.ophys.plotting.experiment_summary_figures as esf
import logging
import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')

def create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True):
    print('saving ', str(experiment_id), 'to', cache_dir)
    dataset = OpenScopePredictiveCodingDataset(experiment_id, cache_dir)
    analysis = ResponseAnalysis(dataset, overwrite_analysis_files)

    # response_df = analysis.get_response_df('oddball')
    # stimulus_table = analysis.dataset.stimulus_table.copy()
    # session_block_names = stimulus_table[stimulus_table.session_block_name.isin(
    #     ['natural_movie_one', 'natural_movie_two']) == False].session_block_name.unique()
    # for session_block_name in session_block_names:
    #     response_df = analysis.get_response_df(session_block_name)

    print('plotting experiment summary figure')
    esf.plot_experiment_summary_figure(analysis, save_dir = dataset.analysis_dir)
    esf.plot_experiment_summary_figure(analysis, save_dir = cache_dir)
    print('plotting cell summary figures')
    for cell_index in dataset.cell_indices:
        sf.plot_cell_summary_figure(analysis, cell_index, save=True, show=True)


if __name__ == '__main__':
    import sys

    experiment_id = sys.argv[1]
    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis'
    create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=False)

    # experiment_ids = [746270939, 746271249,
    #                   750534428, 752473496,
    #                   746271665, 750845430,
    #                   750846019, 752473630,
    #                   746271665, 750845430, 750846019, 752473630,
    #                   755645219, 756118288, 758305436, 759037671]

    # experiment_id = 756118288

    # cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\opc\opc_analysis'
    # create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True)
    # for experiment_id in experiment_ids:
    #     create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True)
