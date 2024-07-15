ARG IMAGE_NAME
FROM ${IMAGE_NAME}:10.1-runtime-centos7 as base

FROM base as base-amd64

ENV NV_CUDNN_VERSION 7.6.5.32-1
ENV NV_CUDNN_PACKAGE libcudnn7-${NV_CUDNN_VERSION}.cuda10.1

ENV NV_CUDNN_DL_HASHCMD sha256sum
ENV NV_CUDNN_DL_SUM e3184cf4c428f457289b2594969575402cd78496f652b7b27abaaaf3a562ae0f
ENV NV_CUDNN_DL_BASENAME libcudnn7-7.6.5.32-1.cuda10.1.x86_64.rpm
ENV NV_CUDNN_DL_URL https://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/libcudnn7-7.6.5.32-1.cuda10.1.x86_64.rpm

RUN yum -y install wget

RUN wget -q ${NV_CUDNN_DL_URL} \
    && echo "${NV_CUDNN_DL_SUM}  ${NV_CUDNN_DL_BASENAME}" | ${NV_CUDNN_DL_HASHCMD} -c - \
    && rpm -i ${NV_CUDNN_DL_BASENAME} \
    && rm -f ${NV_CUDNN_DL_BASENAME} \
    && yum -y remove wget


FROM base as base-ppc64le

ENV NV_CUDNN_VERSION 7.6.5.32-1
ENV NV_CUDNN_PACKAGE libcudnn7-${NV_CUDNN_VERSION}.cuda10.1

ENV NV_CUDNN_DL_HASHCMD sha256sum
ENV NV_CUDNN_DL_SUM 76c8f0ba3838c5b16fdf7c4f33a01b02f3ae59cf744d35b81a597a9c47bb951b
ENV NV_CUDNN_DL_BASENAME libcudnn7-7.6.5.32-1.cuda10.1.ppc64le.rpm
ENV NV_CUDNN_DL_URL https://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/ppc64le/libcudnn7-7.6.5.32-1.cuda10.1.ppc64le.rpm

RUN yum -y install wget

RUN wget -q ${NV_CUDNN_DL_URL} \
    && echo "${NV_CUDNN_DL_SUM}  ${NV_CUDNN_DL_BASENAME}" | ${NV_CUDNN_DL_HASHCMD} -c - \
    && rpm -i ${NV_CUDNN_DL_BASENAME} \
    && rm -f ${NV_CUDNN_DL_BASENAME} \
    && yum -y remove wget


FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <sw-cuda-installer@nvidia.com>"

ENV CUDNN_VERSION ${NV_CUDNN_VERSION}
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN yum install -y \
    ${NV_CUDNN_PACKAGE} \
    && yum clean all \
    && rm -rf /var/cache/yum/*
