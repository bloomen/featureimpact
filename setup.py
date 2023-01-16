#!/usr/bin/env python
from distutils.core import setup
from subprocess import check_call, PIPE
import os
import sys


version = 'VERSION.txt'
if os.path.exists('.git'):
    check_call(['git', 'describe', '--tags', '--dirty=M'],
               stdout=open(version, 'w'), stderr=PIPE)

setup(
    name="featureimpact",
    packages=["featureimpact"],
    version=open(version).read().strip(),
    description="Compute the statistical impact of features given a trained estimator",
    author="Christian Blume",
    author_email="chr.blume@gmail.com",
    url="https://github.com/bloomen/featureimpact",
    license="MIT",
    keywords=["machine learning", "data mining", "statistics"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development"
        ],
    long_description=open('README.md').read(),
    install_requires=['numpy', 'scipy', 'pandas'],
)
