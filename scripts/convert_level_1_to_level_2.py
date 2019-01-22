#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

from openscope_predictive_coding.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2


if __name__ == '__main__':
    import sys
    experiment_id = sys.argv[1]
    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis'
    ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir)
