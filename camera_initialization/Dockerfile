# Use an official ROS2 base image
FROM osrf/ros:humble-desktop

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    usbutils \
    build-essential \
    cmake \
    git \
    ros-humble-rqt \
    ros-humble-rqt-image-view \
    ros-humble-usb-cam \
    entr \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --default-timeout=1000 opencv-python-headless numpy

# Copy your ROS2 workspace
COPY . /root/ros2_ws

# Copy the watch and build script
COPY watch_and_build.sh /root/watch_and_build.sh

# Install Cyclone DDS
RUN apt update && \
    apt install -y ros-humble-rmw-cyclonedds-cpp

# Set the RMW implementation to Cyclone DDS
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Build the workspace and log the output to console
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && cd /root/ros2_ws && colcon build --event-handlers console_cohesion+"

# Copy the camera configuration files
COPY params_1.yaml /opt/ros/humble/share/usb_cam/config/params_1.yaml
COPY params_2.yaml /opt/ros/humble/share/usb_cam/config/params_2.yaml
COPY camera_info.yaml /opt/ros/humble/share/usb_cam/config/camera_info.yaml

# Source the setup file and set RMW implementation in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc

# Source the setup file and list installed packages
RUN /bin/bash -c "source /root/ros2_ws/install/setup.bash && ros2 pkg list"

# Make the watch and build script executable
RUN chmod +x /root/watch_and_build.sh

# Set the entrypoint to launch the watch and build script
ENTRYPOINT ["/bin/bash", "-c", "/root/watch_and_build.sh"]