from setuptools import find_packages, setup

package_name = 'neural_network_stereo_depth_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    py_modules=[
        'neural_network_stereo_depth_pkg.neural_network_depth_estimation_node',
        'neural_network_stereo_depth_pkg.hitnet_model',
        'neural_network_stereo_depth_pkg.cre_model'
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='razturgeman',
    maintainer_email='razturgeman@todo.todo',
    description='ROS2 package for neural network stereo depth estimation',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'neural_network_depth_estimation_node = neural_network_stereo_depth_pkg.neural_network_depth_estimation_node:main'
        ],
    },
)
