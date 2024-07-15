ARG IMAGE_NAME
FROM ${IMAGE_NAME}:12.0.1-runtime-ubuntu18.04 as base

ENV NV_CUDA_LIB_VERSION "12.0.1-1"

FROM base as base-amd64

ENV NV_CUDA_CUDART_DEV_VERSION 12.0.146-1
ENV NV_NVML_DEV_VERSION 12.0.140-1
ENV NV_LIBCUSPARSE_DEV_VERSION 12.0.1.140-1
ENV NV_LIBNPP_DEV_VERSION 12.0.1.104-1
ENV NV_LIBNPP_DEV_PACKAGE libnpp-dev-12-0=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_VERSION 12.0.2.224-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME libcublas-dev-12-0
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_NVPROF_VERSION 12.0.146-1
ENV NV_NVPROF_DEV_PACKAGE cuda-nvprof-12-0=${NV_NVPROF_VERSION}

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-dev-12-0=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-12-0=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-0=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-12-0=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-12-0=${NV_NVML_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-12-0=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
