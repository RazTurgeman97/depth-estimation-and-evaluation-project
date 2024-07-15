ARG IMAGE_NAME
FROM ${IMAGE_NAME}:12.3.2-devel-ubuntu20.04 as base

FROM base as base-amd64

ENV NV_CUDNN_VERSION 9.0.0.312
ENV NV_CUDNN_PACKAGE_NAME "libcudnn9-cuda-12"

ENV NV_CUDNN_PACKAGE "libcudnn9-cuda-12=${NV_CUDNN_VERSION}-1"
ENV NV_CUDNN_PACKAGE_DEV "libcudnn9-dev-cuda-12=${NV_CUDNN_VERSION}-1"


FROM base as base-arm64

ENV NV_CUDNN_VERSION 9.0.0.312
ENV NV_CUDNN_PACKAGE_NAME "libcudnn9-cuda-12"

ENV NV_CUDNN_PACKAGE "libcudnn9-cuda-12=${NV_CUDNN_VERSION}-1"
ENV NV_CUDNN_PACKAGE_DEV "libcudnn9-dev-cuda-12=${NV_CUDNN_VERSION}-1"

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*
