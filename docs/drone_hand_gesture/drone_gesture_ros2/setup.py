from setuptools import setup
import os
from glob import glob

package_name = 'drone_gesture_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, f'{package_name}.nodes', f'{package_name}.scripts'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arrebol',
    maintainer_email='arrebol@example.com',
    description='Drone hand gesture control using ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_control_node = drone_gesture_control.nodes.gesture_control_node:main',
            'visualization_node = drone_gesture_control.nodes.visualization_node:main',
            'visualization_node_fixed = drone_gesture_control.nodes.visualization_node_fixed:main',
            'status_monitor_node = drone_gesture_control.nodes.status_monitor_node:main',
            'train_gesture_model = drone_gesture_control.scripts.train_gesture_model:main',
            'collect_gesture_data = drone_gesture_control.scripts.gesture_data_collector:main',
        ],
    },
)
