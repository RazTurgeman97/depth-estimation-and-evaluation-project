#!/usr/bin/env bats

load helpers

image="${IMAGE_NAME}:${CUDA_VERSION}-devel-${OS}${IMAGE_TAG_SUFFIX}"

function setup() {
    check_runtime
}

@test "check_nvcc_installed" {
    docker_run --rm --gpus 0 ${image} bash -c "stat /usr/local/cuda/bin/nvcc"
    [ "$status" -eq 0 ]
    # image cleanup is done in run_tests.sh
}

@test "check_gcc_installed" {
    docker_run --rm --gpus 0 ${image} bash -c "gcc --version"
    [ "$status" -eq 0 ]
    # image cleanup is done in run_tests.sh
}

@test "check_ncu_installed" {
    unsupported=('10.1')
    debug "cuda_version: ${CUDA_VERSION}"
    debug "unsupported: $(printf '%s' \"${unsupported[@]}\")"
    if printf '%s' "${unsupported[@]}" | grep -q "${CUDA_VERSION}"; then
        skip "ncu test is not supported for this CUDA version"
    fi
    docker_run --rm --gpus 0 ${image} bash -c "ncu --version"
    [ "$status" -eq 0 ]
    # image cleanup is done in run_tests.sh
}
