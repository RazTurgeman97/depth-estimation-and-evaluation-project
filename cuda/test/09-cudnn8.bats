#!/usr/bin/env bats

load helpers

image="${IMAGE_NAME}:${CUDA_VERSION}-cudnn8-devel-${OS}${IMAGE_TAG_SUFFIX}"

function setup() {
    run docker pull --platform linux/${ARCH} ${image}
    if [ "$status" -ne 0 ]; then
        skip "'${image}' does not exist"
    fi
    check_runtime
}

function teardown() {
    docker rmi -f ${image}
}

function check_ubuntu() {
    vers=$(docker run --rm --gpus 0 ${image} bash -c "apt show libcudnn8 2>&1 | grep Version | cut -d: -f2 | xargs")
    [ "$vers" != "" ]
    debug "runtime version: $vers"
    vers_dev=$(docker run --rm --gpus 0 ${image} bash -c "apt show libcudnn8-dev 2>&1 | grep Version | cut -d: -f2 | xargs")
    [ "$vers_dev" != "" ]
    debug "devel version: $vers_dev"
    [ "$vers" = "$vers_dev" ]
}

function check_rhel() {
    vers=$(docker run --rm --gpus 0 ${image} bash -c "rpm -qa | grep libcudnn8 | grep -v devel")
    debug "runtime version: $vers"
    [ "$vers" != "" ]
    # sed chops out the 'devel-' so it's eaiser to compare
    vers_dev=$(docker run --rm --gpus 0 ${image} bash -c "rpm -qa | grep libcudnn8 | grep devel | sed 's/devel-//'")
    [ "$vers_dev" != "" ]
    debug "devel version ('devel-' pre-stripped): $vers_dev"
    [ "$vers" = "$vers_dev" ]
}

@test "check_cudnn8_runtime_package" {
    debug ${image}

    # RUN WITH
    #
    # DEBUG=1
    #
    # set in the enviroment for more helpful output!

    [[ "${OS_NAME}" == "ubuntu" ]] && check_ubuntu && return
    [[ "${OS_NAME}" == "centos" ]] || [[ "${OS_NAME}" == "ubi" ]] || [[ "${OS_NAME}" == "rockylinux" ]] && check_rhel
}
