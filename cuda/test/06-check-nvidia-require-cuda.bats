#!/usr/bin/env bats

load helpers

image="${IMAGE_NAME}:${CUDA_VERSION}-devel-${OS}${IMAGE_TAG_SUFFIX}"

function setup() {
    check_runtime
}

@test "check_NVIDIA_REQUIRES_CUDA" {
    # The devel images for x86_64 should always contain "brand=tesla"
    if [ ${ARCH} != "x86_64" ]; then
       skip "Only needed on x86_64."
    fi
    debug "cuda_version: ${CUDA_VERSION}"
    docker_run --rm --gpus 0 --platform linux/${ARCH} ${image} bash -c "printenv | grep -q 'cuda>='"
    [ "$status" -eq 0 ]
    # image cleanup is done in run_tests.sh
}
