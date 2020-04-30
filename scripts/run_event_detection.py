import sys
import platform
if platform.system() == 'Linux':
    # sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob # flake8: noqa: E999

# pilot study manuscript final expts
experiment_ids = [768898762, 775058863, 775613721, 776727982, 813071318, 816795279,
       817251851, 818894752, 826576489, 827232898, 827235482, 828956958,
       828959377, 829411383, 829417358, 830075254, 830688059, 830688102,
       831312165, 831314921, 832107135, 832601977, 832617299, 833599179,
       833611925, 833612445, 833614835, 833626456, 834244626, 834251985,
       834260382, 835642229, 835654507, 835660148, 836246273, 836248932,
       836250018, 836253258, 836890936, 836891984, 836895367, 836906598,
       837283374, 837285285, 837287590, 837630919, 838330377, 848005700,
       848006710, 848690810, 848691390]


python_file = r"/home/marinag/openscope_predictive_coding/scripts/event_detection.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords'

job_settings = {'queue': 'braintv',
                'mem': '200g',
                'walltime': '40:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

for experiment_id in experiment_ids:
    PythonJob(
        python_file,
        python_executable='/home/marinag/anaconda2/envs/visual_behavior_sdk/bin/python',
        python_args=experiment_id,
        conda_env=None,
        jobname='process_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)
