import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'cash_rl'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jorge Leon Duran',
    maintainer_email='115505677+leonjorge010@users.noreply.github.com',
    description='Gymnasium environment and training entry points for the C.A.$.H hallway DRL navigation policy',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'random_agent = cash_rl.random_agent:main',
        ],
    },
)
