import sys
import platform
if platform.system() == 'Linux':
    # sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob # flake8: noqa: E999

python_file = r"/home/marinag/openscope_predictive_coding/scripts/create_multi_session_mean_df.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords'

job_settings = {'queue': 'braintv',
                'mem': '60g',
                'walltime': '20:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

PythonJob(
    python_file,
    python_executable='/home/marinag/anaconda2/envs/visual_behavior_sdk/bin/python',
    python_args=None,
    conda_env=None,
    jobname='process_multi_session_df',
    **job_settings
).run(dryrun=False)
