#!/usr/bin/python

from setuptools import setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='diplomacy',
    version='0.1',
    packages=['diplomacy'],
    install_requires=required,
)
