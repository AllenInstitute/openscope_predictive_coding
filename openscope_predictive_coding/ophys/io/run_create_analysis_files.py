import sys
import platform
from pbstools import PythonJob

if platform.system() == 'Linux':
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')

lims_ids = [746271249]

python_file = r"/home/marinag/openscope_predictive_coding/openscope_predictive_coding/ophys/io/create_analysis_files.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords2'

job_settings = {'queue': 'braintv',
                'mem': '60g',
                'walltime': '32:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

for lims_id in lims_ids:
    print(lims_id)
    PythonJob(
        python_file,
        python_executable='/home/marinag/anaconda2/envs/visual_behavior_ophys/bin/python',
        python_args=lims_id,
        conda_env=None,
        jobname='process_{}'.format(lims_id),
        **job_settings
    ).run(dryrun=False)
