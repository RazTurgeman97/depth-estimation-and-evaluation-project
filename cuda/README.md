# Nvidia CUDA Linux Container Image Sources

Usage of the CUDA container images requires the [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit).

Container images are available from:

- https://ngc.nvidia.com/catalog/containers/nvidia:cuda
- https://hub.docker.com/r/nvidia/cuda

## Announcement

### Cuda 12 images are now LIVE

Entrypoint scripts are rolling out now for all images, which include deprecation notices for image sets that have reached End-of-life.

Please read our [Container Tag Support Policy](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/doc/support-policy.md) for more information.

### Centos7 and Ubi7 Deprecation Notice

Centos7 and Ubi7 based images are available for CUDA 12.4.0. Container images starting with CUDA 12.4.1 are no longer produced for these operating systems.

### PowerPC images Deprecation Notice

CUDA 12.4 are the last releases to support PowerPC. Beginning 12.5+ we no longer support ppc64le based images.

### Deprecated: "latest" tag

The "latest" tag for CUDA, CUDAGL, and OPENGL images has been deprecated on NGC and Docker Hub.

With the removal of the latest tag, the following use case will result in the "manifest unknown"
error:

```
$ docker pull nvidia/cuda
Error response from daemon: manifest for nvidia/cuda:latest not found: manifest unknown: manifest
unknown
```

This is not a bug.

## IMAGE SECURITY NOTICE

The CUDA images are scanned for CVE vulnerabilities prior to release and some images may contain CVEs at the time of publication.

Our Product Security teams reviews the CVEs and determines if the CVE should block the release or not. We try to mitigate as much as we can, but since we do not control the upstream base images, some cuda image releases might be impacted.

Please consult the README on the NGC or Docker Hub pages for details.

## LD_LIBRARY_PATH NOTICE

The `LD_LIBRARY_PATH` is set inside the container to legacy nvidia-docker v1 paths that do not exist on newer installations. This is done to maintain compatibility for our partners that still use nvidia-docker v1 and this will not be changed for the forseable future. [There is a chance this might cause issues for some.](https://gitlab.com/nvidia/container-images/cuda/-/issues/47)

## Building from source

The container image scripts are archived in the `dist/` directory and are available for all supported distros and cuda versions.

Here is an example on how to build an multi-arch container image for Ubuntu 18.04 and CUDA 11.6.0:

WARNING: cudgl image builds *REQUIRE* a secure registry to push built intermediate images to since buildkit does not easily allow using local image references from the build container.

```bash
./build.sh -d --image-name my-remote-container-registry/cuda --cuda-version 11.6.0 --os ubuntu --os-version 18.04 --arch x86_64,arm64 --push
```

See `./build.sh --help` for usage.

## Cuda Container Image Automation

The [README_CICD.md](https://gitlab.com/nvidia/container-images/cuda/blob/master/README_CICD.md) document provides details on how the gitlab pipelines work and how to control, modify, or debug them.
