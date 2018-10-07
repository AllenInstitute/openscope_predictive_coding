import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob # flake8: noqa: E999

lims_ids =  [746270939, 746271249, 750534428, 752473496, 755645715,
             754579284, 755000515, 755646041, 756118440,
             746271665, 750845430, 750846019, 752473630,
             755645219, 756118288, 758305436, 759037671]

python_file = r"/home/marinag/openscope_predictive_coding/openscope_predictive_coding/ophys/io/convert_level_1_to_level_2.py"

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
