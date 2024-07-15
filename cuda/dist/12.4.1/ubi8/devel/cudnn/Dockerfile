ARG IMAGE_NAME
FROM ${IMAGE_NAME}:12.4.1-devel-ubi8 as base

FROM base as base-amd64

ENV NV_CUDNN_VERSION 9.1.0.70-1
ENV NV_CUDNN_PACKAGE libcudnn9-cuda-12-${NV_CUDNN_VERSION}
ENV NV_CUDNN_PACKAGE_DEV libcudnn9-devel-cuda-12-${NV_CUDNN_VERSION}

FROM base as base-arm64

ENV NV_CUDNN_VERSION 9.1.0.70-1
ENV NV_CUDNN_PACKAGE libcudnn9-cuda-12-${NV_CUDNN_VERSION}
ENV NV_CUDNN_PACKAGE_DEV libcudnn9-devel-cuda-12-${NV_CUDNN_VERSION}


FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <sw-cuda-installer@nvidia.com>"

LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN yum install -y \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && yum clean all \
    && rm -rf /var/cache/yum/*