ARG IMAGE_NAME
FROM ${IMAGE_NAME}:10.0-devel-ubuntu18.04 as base

FROM base as base-amd64

ENV NV_CUDNN_PACKAGE_VERSION 7.6.5.32-1
ENV NV_CUDNN_VERSION 7.6.5.32

ENV NV_CUDNN_PACKAGE_NAME libcudnn7
ENV NV_CUDNN_PACKAGE ${NV_CUDNN_PACKAGE_NAME}=${NV_CUDNN_PACKAGE_VERSION}+cuda10.0
ENV NV_CUDNN_PACKAGE_DEV ${NV_CUDNN_PACKAGE_NAME}-dev=${NV_CUDNN_PACKAGE_VERSION}+cuda10.0
ENV NV_CUDNN_DL_HASHCMD sha256sum
ENV NV_CUDNN_DL_SUM a7e21448181bf69b2f4b9fb3e8c8bdfd962e327b1716d34608488641b4199bb2
ENV NV_CUDNN_DL_BASENAME libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb
ENV NV_CUDNN_DL_URL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb

RUN apt-get update && apt-get install -y --no-install-recommends wget

RUN wget -q ${NV_CUDNN_DL_URL} \
    && echo "${NV_CUDNN_DL_SUM}  ${NV_CUDNN_DL_BASENAME}" | ${NV_CUDNN_DL_HASHCMD} -c - \
    && dpkg -i ${NV_CUDNN_DL_BASENAME} \
    && rm -f ${NV_CUDNN_DL_BASENAME}

ENV NV_CUDNN_DEV_DL_HASHCMD sha256sum
ENV NV_CUDNN_DEV_DL_SUM 2a9a15089040ea53c5ed224f39a5d9d43319d6db7c499716e7d222acf3cb3427
ENV NV_CUDNN_DEV_DL_BASENAME libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb
ENV NV_CUDNN_DEV_DL_URL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb

RUN wget -q ${NV_CUDNN_DEV_DL_URL} \
    && echo "${NV_CUDNN_DEV_DL_SUM}  ${NV_CUDNN_DEV_DL_BASENAME}" | ${NV_CUDNN_DEV_DL_HASHCMD} -c - \
    && dpkg -i ${NV_CUDNN_DEV_DL_BASENAME} \
    && rm -f ${NV_CUDNN_DEV_DL_BASENAME} \
    && apt-get purge --autoremove -y wget


FROM base as base-ppc64le

ENV NV_CUDNN_PACKAGE_VERSION 7.6.5.32-1
ENV NV_CUDNN_VERSION 7.6.5.32

ENV NV_CUDNN_PACKAGE_NAME libcudnn7
ENV NV_CUDNN_PACKAGE ${NV_CUDNN_PACKAGE_NAME}=${NV_CUDNN_PACKAGE_VERSION}+cuda10.0
ENV NV_CUDNN_PACKAGE_DEV ${NV_CUDNN_PACKAGE_NAME}-dev=${NV_CUDNN_PACKAGE_VERSION}+cuda10.0
ENV NV_CUDNN_DL_HASHCMD sha256sum
ENV NV_CUDNN_DL_SUM 1e650f9e87ef160730fadd909fcd511be633d7bf07761183bfcfd839239e033d
ENV NV_CUDNN_DL_BASENAME libcudnn7_7.6.5.32-1+cuda10.0_ppc64el.deb
ENV NV_CUDNN_DL_URL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/ppc64el/libcudnn7_7.6.5.32-1+cuda10.0_ppc64el.deb

RUN apt-get update && apt-get install -y --no-install-recommends wget

RUN wget -q ${NV_CUDNN_DL_URL} \
    && echo "${NV_CUDNN_DL_SUM}  ${NV_CUDNN_DL_BASENAME}" | ${NV_CUDNN_DL_HASHCMD} -c - \
    && dpkg -i ${NV_CUDNN_DL_BASENAME} \
    && rm -f ${NV_CUDNN_DL_BASENAME}

ENV NV_CUDNN_DEV_DL_HASHCMD sha256sum
ENV NV_CUDNN_DEV_DL_SUM 3c53602b5a9e6fd7405a0598a8bbe952a68880a56beeee09dc427f79388defa0
ENV NV_CUDNN_DEV_DL_BASENAME libcudnn7-dev_7.6.5.32-1+cuda10.0_ppc64el.deb
ENV NV_CUDNN_DEV_DL_URL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/ppc64el/libcudnn7-dev_7.6.5.32-1+cuda10.0_ppc64el.deb

RUN wget -q ${NV_CUDNN_DEV_DL_URL} \
    && echo "${NV_CUDNN_DEV_DL_SUM}  ${NV_CUDNN_DEV_DL_BASENAME}" | ${NV_CUDNN_DEV_DL_HASHCMD} -c - \
    && dpkg -i ${NV_CUDNN_DEV_DL_BASENAME} \
    && rm -f ${NV_CUDNN_DEV_DL_BASENAME} \
    && apt-get purge --autoremove -y wget


FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV CUDNN_VERSION ${NV_CUDNN_VERSION}

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} && \
    rm -rf /var/lib/apt/lists/*