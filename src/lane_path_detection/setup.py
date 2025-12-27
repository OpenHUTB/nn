from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'lane_path_detection'

setup(
    name=package_name,
    version='0.0.0',
    # 自动查找所有子模块（适配你的目录结构）
    packages=find_packages(exclude=['test']),
    # 声明源码目录为当前目录
    package_dir={'': '.'},
    # 资源文件配置（保留你的原有逻辑）
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), glob('resource/*')),
    ],
    # ROS2节点入口点（关键：关联main.py的main函数）
    entry_points={
        'console_scripts': [
            'lane_path_detection_node = lane_path_detection.main:main',
        ],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rosli',
    maintainer_email='rosli@todo.todo',
    description='Lane path detection node for ROS2',
    license='Apache-2.0',
    tests_require=['pytest'],
)
