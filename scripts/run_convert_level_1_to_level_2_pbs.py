import os
import sys
import platform
import pandas as pd
if platform.system() == 'Linux':
    # sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob # flake8: noqa: E999


cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis'
manifest = pd.read_excel(os.path.join(cache_dir, 'opc_production_manifest.xlsx'))
experiment_ids = manifest['experiment_id'].values

python_file = r"/home/marinag/openscope_predictive_coding/scripts/convert_level_1_to_level_2.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords_opc'

job_settings = {'queue': 'braintv',
                'mem': '10g',
                'walltime': '5:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

for experiment_id in experiment_ids:
    print(experiment_id)
    PythonJob(
        python_file,
        python_executable='/home/marinag/anaconda2/envs/visual_behavior_ophys/bin/python',
        python_args=experiment_id,
        conda_env=None,
        jobname='process_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)
