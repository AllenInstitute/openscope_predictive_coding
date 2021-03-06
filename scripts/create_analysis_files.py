#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

from openscope_predictive_coding.ophys.io.create_analysis_files import create_analysis_files



if __name__ == '__main__':
    import sys
    experiment_id = sys.argv[1]
    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis'
    create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True, preload_response_dfs=True, use_events=True)
