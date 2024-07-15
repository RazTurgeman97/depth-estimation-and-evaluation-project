#!/bin/bash

# Source ROS 2 setup
source /opt/ros/humble/setup.bash

# Source the workspace setup
source /root/ros2_ws/install/setup.bash

# Initial build
colcon build --event-handlers console_cohesion+

# Watch for changes and rebuild
find /root/ros2_ws/src -name '*.py' | entr -r bash -c "source /opt/ros/humble/setup.bash && source /root/ros2_ws/install/setup.bash && colcon build --event-handlers console_cohesion+ && ros2 launch camera_initialization camera_and_image_splitter_launch.py"

