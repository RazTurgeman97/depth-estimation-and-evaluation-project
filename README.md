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

## Startup command execution

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


## If you get "chown: cannot access '/var/run/docker.sock': No such file or directory"

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
