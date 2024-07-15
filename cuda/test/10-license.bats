#!/usr/bin/env bats

load helpers

image="${IMAGE_NAME}:${CUDA_VERSION}-devel-${OS}${IMAGE_TAG_SUFFIX}"

license_pdf="https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf"
license_pdf_sha1="346e0c3179c7f665f2b18476ecbf03b88e3d93b7"

license_txt_sha1="adf6832428448b1ee9bd77e092ac55416f53450a"

function setup() {
    check_runtime
}

@test "check_license_changed_upstream" {
    #  If this test is failing, the text of the license PDF needs to be converted to text and placed into the license file in this repository.
    #  Once that is done, all supported containers must be rebuilt to contain this new license.
    docker_run --rm --gpus 0 --platform linux/${ARCH} almalinux:9 bash -c "curl -fsSLO ${license_pdf} && echo "${license_pdf_sha1}  NVIDIA_Deep_Learning_Container_License.pdf" | sha1sum -c --strict -"
    [ "$status" -eq 0 ]
    # image cleanup is done in run_tests.h
}

@test "check_license_is_latest_in_container" {
    docker_run --rm --gpus 0 --platform linux/${ARCH} ${image} bash -c "echo "${license_txt_sha1}  /NGC-DL-CONTAINER-LICENSE" | sha1sum -c --strict -"
    [ "$status" -eq 0 ]
    # image cleanup is done in run_tests.sh
}
