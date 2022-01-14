#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:23:04 2020

@author: yoonahshin
"""

from setuptools import setup

setup(name='finger_analysis',
      version='0.1',
      description='Useful tools for analyzing finger morphology',
      author='Yoon Ah Shin',
      author_email='yoonah@mit.edu',
      license='GNU',
      packages=['finger_analysis'],
      install_requires=[
            'pandas',
            'numpy',
            'scikit-image',
            'opencv-python',
            'matplotlib',
            'pytesseract',
            'seaborn',
            ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
