#!/usr/bin/env python
from distutils.core import setup
from subprocess import check_call
import tempfile


def get_version():
    version = tempfile.TemporaryFile()
    check_call(['git', 'describe', '--tags', '--dirty=M'], stdout=version)
    version.seek(0)
    return version.read().strip()


setup(
    name="featureimpact",
    packages=["featureimpact"],
    version=get_version(),
    description="Compute the statistical impact of features given a scikit-learn estimator",
    author="Christian Blume",
    author_email="chr.blume@gmail.com",
    url="http://sourceforge.net/projects/featureimpact",
    license="MIT",
    keywords=["machine learning", "statistics", "scikit-learn"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development"
        ],
    long_description=file('README.txt').read(),
    install_requires=['numpy>=1.6.1',
                      'scipy>=0.9.0',
                      'scikit-learn>=0.13']
)
