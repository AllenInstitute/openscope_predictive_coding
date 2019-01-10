import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob # flake8: noqa: E999

# lims_ids = [746270939, 746271249, 750534428, 752473496,
#              755645715, 754579284, 755000515, 755646041,
#              756118440, 746271665, 750845430, 750846019,
#              752473630, 755645219, 756118288, 758305436,
#              759037671, # pilot2
lims_ids = [768898762, 775613721, 775058863, 776727982, # production
             770649677, 773454193, 774438447, 773914563]


python_file = r"/home/marinag/visual_behavior_analysis/scripts/create_analysis_files.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords_opc'

job_settings = {'queue': 'braintv',
                'mem': '30g',
                'walltime': '20:00:00',
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
