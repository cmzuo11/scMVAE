#!/usr/bin/env python
"""
# Author: cmzuo
# Created Time : Jan 27 Oct 2021 02:42:37 PM CST
# File Name: set_up.py
# Description:
"""
from setuptools import setup, find_packages

with open('used_package.txt') as f:
    requirements = f.read().splitlines()

setup(name='scMVAE',
      version='1.0.1',
      packages=find_packages(),
      description='Deep joint-leaning single-cell multi-omics model',
      long_description='',

      author='Chunman Zuo',
      author_email='',
      url='https://github.com/cmzuo11/scMVAE',
      scripts=['MVAE_test_Adbrain.py'],
      install_requires=requirements,
      python_requires='>3.6.12',
      license='MIT',

      classifiers=[
          'Development Status :: 1 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Bioinformatics',
     ],
     )



