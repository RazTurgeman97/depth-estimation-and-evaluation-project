# Depth Estimation Project
My Depth Estimation Project for my B.S.c Degree in Mechanical Engineering at Ben-Gurion University

This project uses stereo cameras with ROS2 to perform depth estimation.
The following instructions will help you set up the environment using Docker.

## Enable nvidia runtime after reboot:

If you already conducted the instruction below and for some reason you do not want to create a custom systemd service, run the following commands and you are good to go.

```bash
sudo systemctl stop docker.socket

sudo systemctl start docker

docker info | grep -i runtime
```


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
### Startup command execution

to make sure you are good to go without any command to perform you can create a custom systemd service to execute these commands automatically on system reboot.

#### 1. Create a new systemd service file:

```bash
sudo nano /etc/systemd/system/docker-nvidia-runtime.service
```
Add the following content to the file:

```ini

[Unit]
Description=Restart Docker with NVIDIA runtime
After=network.target docker.service

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'systemctl stop docker && systemctl stop docker.socket && systemctl start docker && systemctl restart docker'

[Install]
WantedBy=multi-user.target
```
Save and close the file (press Ctrl + X, then Y, then Enter).

#### 2. Reload the systemd daemon to recognize the new service:

```bash
sudo systemctl daemon-reload
```

#### 3. Enable the service to start on boot:

```bash
sudo systemctl enable docker-nvidia-runtime.service
```

Start the service to verify it works:

```bash
sudo systemctl start docker-nvidia-runtime.service
```

Check the status of the service to ensure it's running correctly:

```bash
sudo systemctl status docker-nvidia-runtime.service
```

This custom systemd service will ensure that your Docker runtime is configured with NVIDIA support every time your system reboots.

### Reboot and Verify

Reboot your system:

```bash
sudo reboot
```

After rebooting, check Docker runtime settings again:

```bash
docker info | grep -i runtime
```
