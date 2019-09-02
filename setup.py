#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages


if os.path.exists('README.md'):
    long_description = open('README.md').read()
else:
    long_description = '''A toolkit for resolving chemical SMILES from structural diagrams.'''

setup(
    name='ChemSchematicDiagramExtractor',
    version='0.0.1',
    author='Edward Beard',
    author_email='ejb207@cam.ac.uk',
    license='MIT',
    url='https://github.com/edbeard/csde',
    packages=find_packages(),
    description='A toolkit for resolving chemical SMILES from structural diagrams.',
    long_description=long_description,
    keywords='image-mining mining chemistry cheminformatics OSR structure diagram html xml science scientific',
    zip_safe=False,
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.png', '*rtf', 'srcnn/*.h5'],
    },
    tests_require=['pytest'],
    install_requires=[
        'pillow', 'tesserocr', 'matplotlib==2.2.4', 'scikit-learn', 'scikit-image<0.15', 'numpy',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
)
