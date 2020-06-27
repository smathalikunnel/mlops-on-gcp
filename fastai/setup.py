import os
from setuptools import setup

required=''

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

#with open('requirements.txt') as f:
#    required = f.read().splitlines()

setup(
    name='my_custom_code',
    version='2.1',
    scripts=['predictor.py', 'preprocess.py'],
    packages=['fastai_custom'],
    install_requires=parse_requirements('requirements.txt'),
    include_package_data=True
)
