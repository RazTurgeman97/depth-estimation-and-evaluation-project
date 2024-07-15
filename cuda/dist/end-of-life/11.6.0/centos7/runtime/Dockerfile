ARG IMAGE_NAME
FROM ${IMAGE_NAME}:11.6.0-base-centos7 as base

ENV NV_CUDA_LIB_VERSION 11.6.0-1

FROM base as base-amd64

ENV NV_NVTX_VERSION 11.6.55-1
ENV NV_LIBNPP_VERSION 11.6.0.55-1
ENV NV_LIBNPP_PACKAGE libnpp-11-6-${NV_LIBNPP_VERSION}
ENV NV_LIBCUBLAS_VERSION 11.8.1.74-1
ENV NV_LIBNCCL_PACKAGE_NAME libnccl
ENV NV_LIBNCCL_PACKAGE_VERSION 2.12.10-1
ENV NV_LIBNCCL_VERSION 2.12.10
ENV NCCL_VERSION 2.12.10
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}-${NV_LIBNCCL_PACKAGE_VERSION}+cuda11.6
FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <sw-cuda-installer@nvidia.com>"

RUN yum install -y \
    cuda-libraries-11-6-${NV_CUDA_LIB_VERSION} \
    cuda-nvtx-11-6-${NV_NVTX_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    libcublas-11-6-${NV_LIBCUBLAS_VERSION} \
    ${NV_LIBNCCL_PACKAGE} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# Add entrypoint items
COPY entrypoint.d/ /opt/nvidia/entrypoint.d/
COPY nvidia_entrypoint.sh /opt/nvidia/
ENV NVIDIA_PRODUCT_NAME="CUDA"
ENV NVIDIA_CUDA_END_OF_LIFE=1
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]