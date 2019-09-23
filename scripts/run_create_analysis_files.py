import os
import sys
import platform
import pandas as pd
if platform.system() == 'Linux':
    # sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob # flake8: noqa: E999


cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis'
manifest_file = r"/allen/programs/braintv/workgroups/nc-ophys/opc/opc_analysis/opc_production_manifest.xlsx"
data = pd.read_excel(manifest_file)
data = data[data['experiment_state'] == 'passed']
experiment_ids = data.experiment_id.unique()
experiment_ids = np.asarray([int(expt_id) for expt_id in experiment_ids])

python_file = r"/home/marinag/openscope_predictive_coding/scripts/create_analysis_files.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords_opc'

job_settings = {'queue': 'braintv',
                'mem': '80g',
                'walltime': '20:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

for experiment_id in experiment_ids:
    print(experiment_id)
    PythonJob(
        python_file,
        python_executable='/home/marinag/anaconda2/envs/visual_behavior_sdk/bin/python',
        python_args=int(experiment_id),
        conda_env=None,
        jobname='process_{}'.format(int(experiment_id)),
        **job_settings
    ).run(dryrun=False)
