ARG IMAGE_NAME
FROM ${IMAGE_NAME}:11.7.1-base-ubuntu18.04 as base

ENV NV_CUDA_LIB_VERSION 11.7.1-1

FROM base as base-amd64

ENV NV_NVTX_VERSION 11.7.91-1
ENV NV_LIBNPP_VERSION 11.7.4.75-1
ENV NV_LIBNPP_PACKAGE libnpp-11-7=${NV_LIBNPP_VERSION}
ENV NV_LIBCUSPARSE_VERSION 11.7.4.91-1

ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas-11-7
ENV NV_LIBCUBLAS_VERSION 11.10.3.66-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}

ENV NV_LIBNCCL_PACKAGE_NAME libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION 2.13.4-1
ENV NCCL_VERSION 2.13.4-1
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda11.7

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-11-7=${NV_CUDA_LIB_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    cuda-nvtx-11-7=${NV_NVTX_VERSION} \
    libcusparse-11-7=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

# Add entrypoint items
COPY entrypoint.d/ /opt/nvidia/entrypoint.d/
COPY nvidia_entrypoint.sh /opt/nvidia/
ENV NVIDIA_PRODUCT_NAME="CUDA"
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]