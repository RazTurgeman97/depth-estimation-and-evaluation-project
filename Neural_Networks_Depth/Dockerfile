# Use the official CUDA image from NVIDIA
FROM nvcr.io/nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install essential packages and ROS2 Humble together
RUN apt update && \
    apt install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    apt install -y software-properties-common && \
    add-apt-repository universe && \
    apt update && \
    apt install -y curl && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list && \
    apt update -y && \
    apt upgrade -y && \
    apt install -y ros-humble-desktop usbutils ros-humble-rqt libgtk2.0-dev pkg-config python3-colcon-common-extensions ros-humble-cv-bridge && \
    apt install -y \
        git \
        libsndfile1-dev \
        tesseract-ocr \
        espeak-ng \
        python3 \
        python3-pip \
        ffmpeg \
        tzdata \
        python3-opencv && \
    ln -fs /usr/share/zoneinfo/Asia/Jerusalem /etc/localtime && \
    echo "Asia/Jerusalem" > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    rm -rf /var/lib/apt/lists/* && \
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Install Python packages
RUN pip3 install opencv-python-headless numpy onnxruntime-gpu

# Copy source code
COPY . /root/ros2_ws

# Install Cyclone DDS
RUN apt update && \
    apt install -y ros-humble-rmw-cyclonedds-cpp && \
    rm -rf /var/lib/apt/lists/*

# Set the RMW implementation to Cyclone DDS
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Install additional CUDA packages
RUN apt update && \
    apt install -y --no-install-recommends \
        cuda-12-5 \
        cuda-demo-suite-12-5 \
        cuda-documentation-12-5 \
        cuda-drivers \
        cuda-nsight-12-5 \
        cuda-nsight-systems-12-5 \
        cuda-nvvp-12-5 \
        cuda-runtime-12-5 \
        cuda-toolkit-12-5 \
        cuda-tools-12-5 \
        cuda-visual-tools-12-5 \
        libcudart11.0 \
        libcudnn8 \
        libcudnn8-dev \
        libcudnn8-samples \
        nvidia-cuda-dev \
        nvidia-cuda-gdb \
        nvidia-cuda-toolkit \
        nvidia-cuda-toolkit-doc && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# Build the workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && cd /root/ros2_ws && colcon build --event-handlers console_cohesion+"

# Source the setup file and set RMW implementation in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=97" >> /root/.bashrc && \
    echo "export ROS_LOCALHOST_ONLY=0" >> /root/.bashrc && \
    echo "export CYCLONEDDS_URI=file:///workspaces/depth-estimation-project/Neural_Networks_Depth/src/cyclonedds.xml" >> /root/.bashrc

# Ensure the nvidia-container-toolkit is installed
RUN apt update && \
    apt install -y nvidia-container-toolkit && \
    rm -rf /var/lib/apt/lists/*

# Copy the NVIDIA entrypoint script
COPY nvidia_entrypoint.sh /opt/nvidia/
RUN chmod +x /opt/nvidia/nvidia_entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]

# Set the working directory
WORKDIR /workspace

# Set default command
CMD ["bash"]
