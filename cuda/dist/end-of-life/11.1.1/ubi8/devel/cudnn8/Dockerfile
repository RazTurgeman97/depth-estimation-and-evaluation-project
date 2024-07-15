ARG IMAGE_NAME
FROM ${IMAGE_NAME}:11.1.1-devel-ubi8 as base

FROM base as base-amd64

ENV NV_CUDNN_VERSION 8.0.5.39-1
ENV NV_CUDNN_PACKAGE libcudnn8-${NV_CUDNN_VERSION}.cuda11.1
ENV NV_CUDNN_PACKAGE_DEV libcudnn8-devel-${NV_CUDNN_VERSION}.cuda11.1

FROM base as base-arm64

ENV NV_CUDNN_VERSION 8.0.5.39-1
ENV NV_CUDNN_PACKAGE libcudnn8-${NV_CUDNN_VERSION}.cuda11.1
ENV NV_CUDNN_PACKAGE_DEV libcudnn8-devel-${NV_CUDNN_VERSION}.cuda11.1

FROM base as base-ppc64le

ENV NV_CUDNN_VERSION 8.0.4.30-1
ENV NV_CUDNN_PACKAGE libcudnn8-${NV_CUDNN_VERSION}.cuda11.1
ENV NV_CUDNN_PACKAGE_DEV libcudnn8-devel-${NV_CUDNN_VERSION}.cuda11.1

ENV NV_CUDNN_DL_HASHCMD sha256sum
ENV NV_CUDNN_DL_SUM 387db4f67b7bd41c81644bbbedaf5f5c5453fa4c0876ffabaade75e5c2e0da80
ENV NV_CUDNN_DL_BASENAME libcudnn8-8.0.4.30-1.cuda11.1.ppc64le.rpm
ENV NV_CUDNN_DL_URL https://developer.download.nvidia.com/compute/machine-learning/repos/rhel8/ppc64le/libcudnn8-8.0.4.30-1.cuda11.1.ppc64le.rpm

RUN yum -y install wget

RUN wget -q ${NV_CUDNN_DL_URL} \
    && echo "${NV_CUDNN_DL_SUM}  ${NV_CUDNN_DL_BASENAME}" | ${NV_CUDNN_DL_HASHCMD} -c - \
    && rpm -i ${NV_CUDNN_DL_BASENAME} \
    && rm -f ${NV_CUDNN_DL_BASENAME}

ENV NV_CUDNN_DEV_DL_HASHCMD sha256sum
ENV NV_CUDNN_DEV_DL_SUM 9626e2cd27e9fa878ce549b61af91dadc6a8a96e76f1202b8e229beac9f207b2
ENV NV_CUDNN_DEV_DL_BASENAME libcudnn8-devel-8.0.4.30-1.cuda11.1.ppc64le.rpm
ENV NV_CUDNN_DEV_DL_URL https://developer.download.nvidia.com/compute/machine-learning/repos/rhel8/ppc64le/libcudnn8-devel-8.0.4.30-1.cuda11.1.ppc64le.rpm

RUN wget -q ${NV_CUDNN_DEV_DL_URL} \
    && echo "${NV_CUDNN_DEV_DL_SUM}  ${NV_CUDNN_DEV_DL_BASENAME}" | ${NV_CUDNN_DEV_DL_HASHCMD} -c - \
    && rpm -i ${NV_CUDNN_DEV_DL_BASENAME} \
    && rm -f ${NV_CUDNN_DEV_DL_BASENAME} \
    && yum -y remove wget


FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <sw-cuda-installer@nvidia.com>"

LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN yum install -y \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && yum clean all \
    && rm -rf /var/cache/yum/*