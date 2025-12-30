#!/usr/bin/env python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[],
    package_dir={},
    scripts=[
        'scripts/main.py',
        'scripts/sensors.py',
        'scripts/tracker.py',
        'scripts/utils.py'
    ]
)

setup(**d)
