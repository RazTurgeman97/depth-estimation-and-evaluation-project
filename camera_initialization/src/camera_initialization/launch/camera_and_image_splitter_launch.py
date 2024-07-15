import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    usb_cam_config = os.path.join(
        get_package_share_directory('usb_cam'),
        'config',
        'params_2.yaml'
    )

    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            namespace='usb_cam',
            parameters=[usb_cam_config]
        ),
        Node(
            package='image_splitter',
            executable='image_splitter_node',
            name='image_splitter'
        ),
        # Node(
        #     package='stereo_triangulation',
        #     executable='triangulation_node',
        #     name='triangulation_node'
        # )
    ])
