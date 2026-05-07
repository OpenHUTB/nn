from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# 获取ROS包配置
d = generate_distutils_setup(
    packages=['mujoco_manrun'],  # 包名
    package_dir={'': 'scripts'}   # 源码目录
)

setup(**d)
