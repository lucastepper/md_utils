#!/usr/bin/env python
from setuptools import setup


setup(
    name='md_utils',
    version='0.1',
    description='Small collection of scripts for MD data analysis',
    author='Lucas Tepper',
    author_email='lucas.tepper.91@gmail.com',
          install_requires=[
          'numpy',
          'MDAnalysis',
      ],
    packages=[
        "md_utils"
    ]
     )
