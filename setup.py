
from setuptools import setup, find_packages

setup(
    name = 'openscope_predictive_coding',
    version = '0.1.0',
    # description = """Utility functions and modules that I use frequently""",
    # author = "Nicholas Cain","Marina Garrett","Hannah Choi","Rylan Larsen"
    # author_email = "nicolasc@alleninstitute.org","marinag@alleninstitute.org","hannahc@alleninstitute.org","rylanl@alleninstitute.org", 
    url = 'https://github.com/AllenInstitute/openscope_predictive_coding',
    packages = find_packages(),
    # install_requires=['pandas', 'visual_behavior==0.5.0dev5', 'requests', 'pyyaml'],
    include_package_data=True,
    # setup_requires=['pytest-runner'],
)

