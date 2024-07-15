from setuptools import find_packages, setup

package_name = 'stereo_camera'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='razturgeman',
    maintainer_email='razturgeman@todo.todo',
    description='ROS2 Stereo Camera Disparity Map',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stereo_matcher = stereo_camera.stereo_matcher:main'
        ],
    },
)
