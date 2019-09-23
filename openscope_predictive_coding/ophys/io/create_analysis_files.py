from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

from openscope_predictive_coding.ophys.dataset.openscope_predictive_coding_dataset import OpenScopePredictiveCodingDataset
from openscope_predictive_coding.ophys.response_analysis.response_analysis import ResponseAnalysis
import openscope_predictive_coding.ophys.plotting.summary_figures as sf
import openscope_predictive_coding.ophys.plotting.experiment_summary_figures as esf
import logging


def create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True, preload_response_dfs=False):
    print('saving ', str(experiment_id), 'to', cache_dir)
    experiment_id = int(experiment_id)
    dataset = OpenScopePredictiveCodingDataset(experiment_id, cache_dir)
    analysis = ResponseAnalysis(dataset, overwrite_analysis_files, preload_response_dfs)
    response_dict = analysis.get_response_df_dict()

    print('plotting experiment summary figure')
    esf.plot_experiment_summary_figure(analysis, save_dir = dataset.analysis_dir)
    esf.plot_experiment_summary_figure(analysis, save_dir = cache_dir)
    print('plotting cell summary figures')
    for cell_specimen_id in dataset.cell_specimen_ids:
        sf.plot_cell_summary_figure(analysis, cell_specimen_id, save=True, show=False)


if __name__ == '__main__':
    # import sys
    # experiment_id = sys.argv[1]
    manifest_file = r"\\allen\programs\braintv\workgroups\nc-ophys\opc\opc_analysis\opc_production_manifest.xlsx"
    data = pd.read_excel(manifest_file)
    data = data[data['experiment_state'] == 'passed']
    experiment_ids = data.experiment_id.unique()
    experiment_ids = np.asarray([int(expt_id) for expt_id in experiment_ids])

    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\opc\opc_analysis'
    for experiment_id in experiment_ids:
        create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True, preload_response_dfs=False)
