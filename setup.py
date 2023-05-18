#!/usr/bin/env python
from distutils.core import setup


setup(
    name="featureimpact",
    packages=["featureimpact"],
    version="2.2.2",
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
