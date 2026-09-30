import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'cash_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jorge Leon Duran',
    maintainer_email='115505677+leonjorge010@users.noreply.github.com',
    description='Gazebo hallway simulation and robot model for the C.A.$.H escort robot',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={'console_scripts': []},
)
