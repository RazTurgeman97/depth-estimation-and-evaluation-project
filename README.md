# depth-estimation-project
My Depth Estimation Project for my B.S.c Degree in Mechanical Engineering at Ben-Gurion University

This project uses stereo cameras with ROS2 to perform depth estimation. The following instructions will help you set up the environment using Docker.


## To make the permission changes for /dev/video* persistent across reboots, you can create a udev rule. Here’s how you can do it:
### Create a new udev rule file:

```bash
sudo nano /etc/udev/rules.d/99-video-permissions.rules
```

### Add the udev rules:

```bash
KERNEL=="video[0-9]*", GROUP="video", MODE="0666"
```

Save the file and close the text editor (Ctrl + O to save, Ctrl + X to exit in nano).

### Reload udev rules:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Verify the changes:

```bash
ls -l /dev/video*
```

## To use nvidia runtime:

### make sure that daemon.json contains the following:

```bash
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "/usr/bin/nvidia-container-runtime"
        }
    }
}
```

by running that command:

```bash
cat /etc/docker/daemon.json
```

if the file needs to be edited you can use nano editor or gedit:

```bash
sudo nano /etc/docker/daemon.json
```

```bash
sudo gedit /etc/docker/daemon.json
```

### make sure that config.toml contains the following:

```bash
#accept-nvidia-visible-devices-as-volume-mounts = false
#accept-nvidia-visible-devices-envvar-when-unprivileged = true
disable-require = false
#swarm-resource = "DOCKER_RESOURCE_GPU"
switchover-file = "/etc/nvidia-container-runtime/switch-over"

[nvidia-container-cli]
#debug = "/var/log/nvidia-container-toolkit.log"
environment = []
ldcache = "/etc/ld.so.cache"
ldconfig = "@/sbin/ldconfig"
load-kmods = true
no-cgroups = false
#path = "/usr/bin/nvidia-container-cli"
#root = "/run/nvidia/driver"
user = "root:video"

[nvidia-container-runtime]
debug = "/var/log/nvidia-container-runtime.log"
log-level = "info"
mode = "auto"

[nvidia-container-runtime.modes]

[nvidia-container-runtime.modes.csv]
mount-spec-path = "/etc/nvidia-container-runtime/host-files-for-container.d"

```

if the file needs to be edited you can use nano editor or gedit:

```bash
sudo nano /etc/nvidia-container-runtime/config.toml
```

```bash
sudo gedit /etc/nvidia-container-runtime/config.toml
```

### Reload and restart Docker after editing :

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### Check Docker info:


```bash
sudo docker info | grep -i runtime
```

## Troubleshooting nvidia runtime:

### Check Permissions:

```bash
ls -l /usr/bin/nvidia-container-runtime
```

You should see something like this:

```bash
-rwxr-xr-x 1 root root 3163080 Jul 18  2023 /usr/bin/nvidia-container-runtime
```

If the permissions are not correct, set them to 755:

```bash
sudo chmod 755 /usr/bin/nvidia-container-runtime
```

### Applying changes:

1. ensure the Docker service is stopped.

```bash
sudo systemctl stop docker
```

Now, run the Docker daemon manually to check for configuration errors:


```bash
sudo dockerd --config-file /etc/docker/daemon.json --config-file /etc/docker/daemon-nvidia.json
```

If the Docker daemon is still running manually in your terminal, stop it by pressing Ctrl+C.

2. Start Docker Service.

```bash
sudo systemctl start docker
```

3. Check Docker info to see if the NVIDIA runtime is now the default:

```bash
sudo docker info | grep -i runtime
```

You should see:

```bash
Runtimes: nvidia runc io.containerd.runc.v2
Default Runtime: nvidia
```

### If you get an ERROR: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?

Start Docker Service.

```bash
sudo systemctl start docker
```

 Enable Docker to Start on Boot (Optional)

 ```bash
sudo systemctl enable docker
```

Verify Docker Service Status

 ```bash
sudo systemctl status docker
```

Check Docker info to see if the NVIDIA runtime is now the default:

```bash
sudo docker info | grep -i runtime
```

### Check Docker Group Membership

Ensure your user is part of the docker group. This will allow you to run Docker commands without needing sudo.

```bash
sudo usermod -aG docker $USER
```

### Check Docker Socket Permissions

Verify the permissions on the Docker socket:

```bash
ls -l /var/run/docker.sock
```

The output should look like this:

```bash
srw-rw---- 1 root docker 0 Jul 14 17:11 /var/run/docker.sock
```

If the permissions are not correct, you can adjust them:

```bash
sudo chmod 660 /var/run/docker.sock
sudo chown root:docker /var/run/docker.sock
```

Restart the Docker service to ensure all changes take effect:

```bash
sudo systemctl restart docker
```

Check Docker info again:


```bash
docker info | grep -i runtime
```

### If you get "chown: cannot access '/var/run/docker.sock': No such file or directory"

First, stop the Docker service:

```bash
sudo systemctl stop docker
sudo systemctl stop docker.socket
```

Ensure there are no Docker processes still running:

```bash
ps aux | grep dockerd
```

If any dockerd processes are still running, kill them:

```bash
sudo kill -9 <PID>
```

Replace <PID> with the actual process ID.

Check if the docker.pid file exists and delete it:

```bash
sudo rm -f /var/run/docker.pid
sudo rm -f /run/snap.docker/docker.pid
```

Start the Docker service again:

```bash
sudo systemctl start docker
```

Check Docker info to see if the NVIDIA runtime is now the default:

```bash
docker info | grep -i runtime
```

## To make the changes persist after reboot:

### Enable Docker to start on boot

```bash
sudo systemctl enable docker
```

### Ensure Docker is using the correct configuration file

```bash
sudo nano /lib/systemd/system/docker.service
```

Update the ExecStart line to include the --config-file option:

```bash
ExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock --config-file /etc/docker/daemon.json
```

The modified section should look like this:

```bash
[Service]
Type=notify
ExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock --config-file /etc/docker/daemon.json
ExecReload=/bin/kill -s HUP $MAINPID
TimeoutStartSec=0
RestartSec=2
Restart=always
```

Save and exit.

### Reload and Restart Docker

Reload the systemd daemon to apply the changes:

```bash
sudo systemctl daemon-reload
```

Restart the Docker service:

```bash
sudo systemctl restart docker
```

Enable Docker to start on boot:

```bash
sudo systemctl enable docker
```

### Reboot and Verify

Reboot your system:

```bash
sudo reboot
```

After rebooting, check Docker runtime settings again:

```bash
docker info | grep -i runtime
```

# usb_cam troubleshooting:

While working on the project and after countless uses of usb_cam on the latest version (0.6.0, and 0.8.1 since the 1st of May) the program started to malfunction.
After much searching, I have found that the problem lies in the usb_cam program. Version 0.6.0 and 0.7.0 persist with the issue, version 0.6.0 seems to be fixing it.

Here is how to remove the usb_cam and install the 0.6.0 release:

## Remove the Current usb_cam Package:

```bash
cd ~/ros2_ws/src
rm -rf usb_cam
```

## Clone the usb_cam Repository and Check Out Version 0.6.0:

```bash
git clone https://github.com/ros-drivers/usb_cam.git
cd usb_cam
git checkout tags/0.6.0
```

## Install Dependencies:
Ensure that all dependencies required by the usb_cam package are installed.

```bash
sudo apt install python3-rosdep2 # If not installed already

cd ~/ros2_ws/src
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

## Build Your Workspace:

```bash
colcon build --packages-select usb_cam
```

## Source Your Workspace:
Source your workspace to update your environment with the newly installed usb_cam package.

```bash
source ~/ros2_ws/install/setup.bash
```

## To ensure that your workspace stays on version 0.6.0 of the usb_cam package and does not get updated inadvertently, you can follow these steps:

### Create a New Branch:
Create a new branch from the detached HEAD state you are currently in. This will ensure that your workspace remains on version 0.6.0.

```bash
cd ~/ros2_ws/src/usb_cam
git switch -c stable-0.6.0
```

### Add a .rosinstall File:
Add a .rosinstall file to your workspace that specifies the exact commit or tag to use for the usb_cam package. This ensures that rosinstall will always check out the specified version.

Create a file named .rosinstall in the root of your workspace (~/ros2_ws):

```bash
cd ~/ros2_ws
nano .rosinstall
```
and paste this content:

```yaml

- git:
    local-name: src/usb_cam
    uri: https://github.com/ros-drivers/usb_cam.git
    version: 0.6.0
```

To save: Ctrl+X, y, Entet.

### Use rosinstall to Update the Workspace:
Use rosinstall to ensure the workspace remains consistent with the specified versions.

```bash
sudo apt install python3-rosinstall # If not installed already

cd ~/ros2_ws
rosinstall .
```

### Commit Your Changes:
Commit the .rosinstall file to your version control system (e.g., git) if you are using one, so that the exact state of the workspace can be reproduced.
