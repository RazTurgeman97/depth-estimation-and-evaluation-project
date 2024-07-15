# Container Support Policy

As of Friday October 28th, 2022 we, the Kitmaker Team at Nvidia, responsible for the Nvidia CUDA Repositories and CUDA container images, are formally announcing a new support policy for CUDA Container Images.

Going forward,  
 - Cuda Toolkit images will End-of-Life (EOL) when the driver it shipped with goes EOL.  
    For example, CUDA 11.4.3 shipped with R470. R470 goes EOL Jul 2024, so CUDA 11.4.3 images will no longer be updated starting Jul 2024.  
 - In addition, we will only support the latest point release for each of these toolkit versions.  
    For example when 12.1.1 is released, 12.1.0 is marked EOL  
 - Once a CUDA container image set has gone EOL, we will update the Message of the Day for the image to indicate it has reached EOL. We will also provide notification via a gitlab issue our customers can subscribe to.
  - We are also including support for 2 major CUDA versions at any given time (e.g. 11 & 12 or 12 & 13). i.e. 11 will be marked EOL, after 13 is released. If 11 is already past deletion date based on above conditions, a month will be added to 11's deletion time to avoid disruption and provide ample time for customers to plan the transition. Please refer to the EOL chart for the deletion dates.

 - After a period of Six Months time, the EOL tags **WILL BE DELETED from Docker Hub and Nvidia GPU Cloud (NGC)**. This deletion ensures unsupported tags (and image layers) are not left lying around for customers to continue using after they have long been abandoned.

Please see the [EOL chart](container_tags.pdf) for driver EOL and approx container tags deletion dates.

## Why

Earlier this year, we had to change the CUDA repo signing key. This had a reverberating impact across many parts of the software we support, but the worst and most surprising were the CUDA Container Images. We found many of our customers were still using no longer supported image tags that we had no way to easily update. Since there has never been a formal support policy in place, we had to update all currently supported image sets and revive legacy code to rebuild old unsupported image sets, and in some cases even this was not feasible.

Due to the nature of shipping practically an entire OS distribution in a container image, these old image sets also include critical CVE vulnerabilities and anyone using any container images should perform regular updates as a best practice.

## Disruption potential

Deleting tags is not ideal. Each tag contains unique "image layers" that when removed can break any image that uses that layer. "Break" in this context means standard container pull commands will fail due to missing image layers leaving users with a not-so-nice error message from the client. Holding onto these image layers is also not sustainable. CUDA is an extremely large software project, and we have not been great at cleaning house which has resulted in our systems having to build and push multiple terabytes of container images every month for tags that barely get any downloads from the public repositories we push to.

We feel once our customers get used to this new support policy, disruptions will reduce to minimum and we can focus our efforts on image sets and tags that actually get used.

## When

We are announcing the policy change today, but it will not get activated until the next CUDA toolkit major version is released. Then six months after that we will start removing tags.

Also, expect a formal blog post announcing the changes soon.

## Example Support Table

Image sets that would remain active

| Image       | Active (recently updated) | Driver            |
|-------------|---------------------------|-------------------|
| Cuda 11.5.2 | Yes                       | r470              |
| Cuda 11.6.2 | Yes                       | r470              |
| Cuda 11.7.1 | Yes                       | r470, r510, r525  |
| Cuda 11.8.0 | Yes                       | r470, r510, r525  |

Image sets that would be sunsetted (and eventually deleted)

| Image               | Active (recently updated) | Driver           |
|---------------------|---------------------------|------------------|
| Cuda 6.5            | No                        | Prehistoric      |
| Cuda 7.0            | No                        | Prehistoric      |
| Cuda 8.0            | No (both)                 | Prehistoric      |
| Cuda, CudaGL 9.0    | No (both)                 | Prehistoric      |
| Cuda, CudaGL 9.1    | No (both)                 | Prehistoric      |
| Cuda, CudaGL 9.2    | No (both)                 | Prehistoric      |
| Cuda, CudaGL 10.0   | No (both)                 | EOL              |
| Cuda, CudaGL 10.1   | Yes, No (CudaGL)          | r418             |
| Cuda, CudaGL 10.2   | Yes, No (CudaGL)          | r418             |
| Cuda, CudaGL 11.0.3 | Yes, No (CudaGL)          | r418, r450       |
| Cuda, CudaGL 11.1.1 | Yes, No (CudaGL)          | r418, r450       |
| Cuda, CudaGL 11.2.2 | Yes, No (CudaGL)          | r418, r450       |
| Cuda, CudaGL 11.2.0 | Yes, No (CudaGL)          | r418, r450       |
| Cuda, CudaGL 11.2.1 | Yes, No (CudaGL)          | r418, r450       |
| Cuda, CudaGL 11.3.0 | Yes, No (CudaGL)          | r418, r450       |
| Cuda, CudaGL 11.3.1 | Yes, No (CudaGL)          | r418, r450       |
| Cuda, CudaGL 11.4.0 | Yes, No (CudaGL)          | r418, r450       |
| Cuda, CudaGL 11.4.1 | Yes, No (CudaGL)          | r418, r450       |
| Cuda, CudaGL 11.4.2 | Yes, No (CudaGL)          | r418, r450       |
| Cuda 11.4.3         | Yes                       | r418, r450       |
| Cuda 11.5.0         | Yes                       | r418, r450, r470 |
| Cuda 11.5.1         | Yes                       | r418, r450, r470 |
| Cuda 11.5.2         | Yes                       | r418, r450, r470 |
| Cuda 11.6.0         | Yes                       | r418, r450, r470 |
| Cuda 11.6.1         | Yes                       | r418, r450, r470 |
| Cuda 11.7.0         | Yes                       | r450, r470, r510 |

## Workarounds

It is possible to build the images that are needed from the container scripts provided using the build script.

Please see the [project readme for instructions](https://gitlab.com/nvidia/container-images/cuda#building-from-source).

## How to get notified of upcoming changes

Please consider subscribing to [Gitlab Issue #206](https://gitlab.com/nvidia/container-images/cuda/-/issues/206) to be notified of breaking changes.

This issue is locked in the Gitlab issue tracker and comments can only be left by Nvidia employees.

## Comments or suggestions

To provide feed back or comments, please goto:

https://gitlab.com/nvidia/container-images/cuda/-/issues/177
