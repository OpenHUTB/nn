from setuptools import setup
from glob import glob
import os

package_name = 'mujoco_ros_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 包含 launch 文件
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.py')),
        # 包含配置文件和资源文件
        (os.path.join('share', package_name, 'config'), 
         glob('config/*.xml')),
        (os.path.join('share', package_name, 'config', 'assets'),
         glob('config/assets/*')),
    ],

    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lbxlb',
    maintainer_email='3378337402@qq.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mujoco_pub = mujoco_ros_demo.mujoco_publisher:main',
            'data_sub = mujoco_ros_demo.data_subscriber:main',
        ],
    },
)
