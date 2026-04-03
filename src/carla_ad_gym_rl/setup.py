from setuptools import setup, find_packages

setup(
      name='easycarla',
      version='0.1.0',
      description='A simple and easy-to-use OpenAI Gym environment based on the CARLA simulator.',
      author='SilverWings',
      packages=find_packages(),
      install_requires=['gym'] 
)