FROM osrf/ros:humble-desktop

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    usbutils \
    ros-humble-rqt \
    ros-humble-rqt-image-view \
    libgtk2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install opencv-python-headless numpy

# Copy calibration file
COPY calibration-camchain.yaml /opt/ros/humble/share/stereo_triangulation/config/calibration-camchain.yaml

# Copy source code
COPY . /root/ros2_ws

# Install Cyclone DDS
RUN apt update && \
    apt install -y ros-humble-rmw-cyclonedds-cpp

# Set the RMW implementation to Cyclone DDS
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Build the workspace and log the output to console
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && cd /root/ros2_ws && colcon build --event-handlers console_cohesion+"

# Source the setup file and set RMW implementation in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=97" >> /root/.bashrc && \
    echo "export ROS_LOCALHOST_ONLY=0" >> /root/.bashrc && \
    echo "export CYCLONEDDS_URI=file:///workspaces/depth-estimation-project/stereo_triangulation/src/cyclonedds.xml" >> /root/.bashrc

# Source the setup file and list installed packages
RUN /bin/bash -c "source /root/ros2_ws/install/setup.bash && ros2 pkg list"

# Create entrypoint script
RUN echo "#!/bin/bash\nsource /opt/ros/humble/setup.bash\nsource /root/ros2_ws/install/setup.bash\nros2 run stereo_triangulation triangulation_node" > /root/entrypoint.sh

# Make entrypoint script executable
RUN chmod +x /root/entrypoint.sh

# Set the entrypoint to the script
ENTRYPOINT ["/root/entrypoint.sh"]
