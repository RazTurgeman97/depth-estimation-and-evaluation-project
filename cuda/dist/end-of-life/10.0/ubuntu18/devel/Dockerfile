ARG IMAGE_NAME
FROM ${IMAGE_NAME}:10.0-runtime-ubuntu18.04 as base

FROM base as base-amd64

ENV NV_CUDA_LIB_VERSION 10.0.130-1
ENV NV_CUDA_CUDART_DEV_VERSION 10.0.130-1
ENV NV_NVML_DEV_VERSION 10.0.130-1
ENV NV_LIBCUSPARSE_DEV_VERSION 10.0.130-1
ENV NV_LIBNPP_DEV_VERSION 10.0.130-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME cuda-cublas-dev-10-0

ENV NV_LIBCUBLAS_DEV_VERSION 10.0.130-1
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-dev
ENV NV_LIBNCCL_DEV_VERSION 2.6.4-1
ENV NCCL_VERSION ${NV_LIBNCCL_DEV_VERSION}
ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_VERSION}+cuda10.0
ENV NV_LIBNCCL_PACKAGE_SHA256SUM f92759e2ab0e36a1c9887f080f7283a774a7c971e352203db4d94d99c44f1028
ENV NV_LIBNCCL_PACKAGE_SOURCE https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnccl2_2.6.4-1+cuda10.0_amd64.deb
ENV NV_LIBNCCL_PACKAGE_SOURCE_NAME libnccl2_2.6.4-1+cuda10.0_amd64.deb

RUN apt-get update && apt-get install -y --no-install-recommends wget

RUN wget -q ${NV_LIBNCCL_PACKAGE_SOURCE} \
    && echo "$NV_LIBNCCL_PACKAGE_SHA256SUM  ${NV_LIBNCCL_PACKAGE_SOURCE_NAME}" | sha256sum -c --strict - \
    && dpkg -i ${NV_LIBNCCL_PACKAGE_SOURCE_NAME} \
    && rm -f ${NV_LIBNCCL_PACKAGE_SOURCE_NAME} \
    && apt-get purge --autoremove -y wget \
    && rm -rf /var/lib/apt/lists/*

ENV NV_LIBNCCL_DEV_PACKAGE_SHA256SUM 5df80a94bc62835d5bec16fa0a510e337e56a43f1b2f1ec8f27278dd22bef0e9
ENV NV_LIBNCCL_DEV_PACKAGE_SOURCE https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnccl-dev_2.6.4-1+cuda10.0_amd64.deb
ENV NV_LIBNCCL_DEV_PACKAGE_SOURCE_NAME libnccl-dev_2.6.4-1+cuda10.0_amd64.deb
RUN apt-get update && apt-get install -y --no-install-recommends wget

RUN wget -q ${NV_LIBNCCL_DEV_PACKAGE_SOURCE} \
    && echo "$NV_LIBNCCL_DEV_PACKAGE_SHA256SUM  ${NV_LIBNCCL_DEV_PACKAGE_SOURCE_NAME}" | sha256sum -c --strict - \
    && dpkg -i ${NV_LIBNCCL_DEV_PACKAGE_SOURCE_NAME} \
    && rm -f ${NV_LIBNCCL_DEV_PACKAGE_SOURCE_NAME} \
    && apt-get purge --autoremove -y wget \
    && rm -rf /var/lib/apt/lists/*

FROM base as base-ppc64le

ENV NV_CUDA_LIB_VERSION 10.0.130-1
ENV NV_CUDA_CUDART_DEV_VERSION 10.0.130-1
ENV NV_NVML_DEV_VERSION 10.0.130-1
ENV NV_LIBCUSPARSE_DEV_VERSION 10.0.130-1
ENV NV_LIBNPP_DEV_VERSION 10.0.130-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME cuda-cublas-dev-10-0
ENV NV_LIBCUBLAS_DEV_VERSION 10.0.130-1
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-dev
ENV NV_LIBNCCL_DEV_VERSION 2.6.4-1
ENV NCCL_VERSION ${NV_LIBNCCL_DEV_VERSION}
ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_VERSION}+cuda10.0
ENV NV_LIBNCCL_PACKAGE_SHA256SUM be3fc521c0b10498bdcff17b8331c71f79e7e17372e8572cc97db3162951955a
ENV NV_LIBNCCL_PACKAGE_SOURCE https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/ppc64el/libnccl2_2.6.4-1+cuda10.0_ppc64el.deb
ENV NV_LIBNCCL_PACKAGE_SOURCE_NAME libnccl2_2.6.4-1+cuda10.0_ppc64el.deb

RUN apt-get update && apt-get install -y --no-install-recommends wget

RUN wget -q ${NV_LIBNCCL_PACKAGE_SOURCE} \
    && echo "$NV_LIBNCCL_PACKAGE_SHA256SUM  ${NV_LIBNCCL_PACKAGE_SOURCE_NAME}" | sha256sum -c --strict - \
    && dpkg -i ${NV_LIBNCCL_PACKAGE_SOURCE_NAME} \
    && rm -f ${NV_LIBNCCL_PACKAGE_SOURCE_NAME} \
    && apt-get purge --autoremove -y wget \
    && rm -rf /var/lib/apt/lists/*

ENV NV_LIBNCCL_DEV_PACKAGE_SHA256SUM 63bf5de89c9c42fc7ef0a0814f8a59fd99ff8a184c1aa191d4d1d340f6704faf
ENV NV_LIBNCCL_DEV_PACKAGE_SOURCE https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/ppc64el/libnccl-dev_2.6.4-1+cuda10.0_ppc64el.deb
ENV NV_LIBNCCL_DEV_PACKAGE_SOURCE_NAME libnccl-dev_2.6.4-1+cuda10.0_ppc64el.deb
RUN apt-get update && apt-get install -y --no-install-recommends wget

RUN wget -q ${NV_LIBNCCL_DEV_PACKAGE_SOURCE} \
    && echo "$NV_LIBNCCL_DEV_PACKAGE_SHA256SUM  ${NV_LIBNCCL_DEV_PACKAGE_SOURCE_NAME}" | sha256sum -c --strict - \
    && dpkg -i ${NV_LIBNCCL_DEV_PACKAGE_SOURCE_NAME} \
    && rm -f ${NV_LIBNCCL_DEV_PACKAGE_SOURCE_NAME} \
    && apt-get purge --autoremove -y wget \
    && rm -rf /var/lib/apt/lists/*

FROM base-${TARGETARCH}

ARG TARGETARCH
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvml-dev-10-0=${NV_NVML_DEV_VERSION} \
    cuda-command-line-tools-10-0=${NV_CUDA_LIB_VERSION} \
    cuda-nvprof-10-0=${NV_CUDA_LIB_VERSION} \
    cuda-npp-dev-10-0=${NV_LIBNPP_DEV_VERSION} \
    cuda-libraries-dev-10-0=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-10-0=${NV_CUDA_LIB_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# apt from auto upgrading the cublas package. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs