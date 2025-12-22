from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=[],
    package_dir={"": "scripts"},
    scripts=["scripts/main.py"] 
)

setup(**setup_args)
