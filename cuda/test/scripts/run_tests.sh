#!/bin/bash

set -x
set -e

image="${IMAGE_NAME}:${CUDA_VERSION}-devel-${OS}${IMAGE_TAG_SUFFIX}"

docker pull --platform linux/${ARCH} ${image}
docker pull --platform linux/${ARCH} almalinux:9

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
test_path=$(realpath "${script_dir}/../")

for test in $(find $test_path -iname "[0-9]*-*.bats" | sort); do
  if [[ "${test}" == *nvidia-smi* ]] && [[ "${ARCH}" != "x86_64" ]]; then
      # We are running tests on an x86_64 machine for multi-arch and nvidia-smi is passed to the
      # container from the host so it won't work on anything but x86_64.
      echo "Skipping test '${test}' on architecture ${ARCH}"
      continue
  fi
  if [[ "${test}" == *samples* ]] && [[ "${ARCH}" != "x86_64" ]]; then
      # FIXME: Samples tests on multi-arch
      echo "Skipping test '${test}' on architecture ${ARCH}"
      continue
  fi
  if [[ "${test}" == *license* ]] && [[ "${ARCH}" != "ppc64el" ]]; then
      # FIXME: use a ppc64le container with curl
      echo "Skipping test '${test}' on architecture ${ARCH}"
      continue
  fi
  if [[ "${test}" == *"test/08-cudnn7.bats"* ]] || [[ "${test}" == *"test/09-cudnn8.bats"* ]]; then
      # 23:07 Sat May 21 2022: temporary disable until server resources can be increased
      echo "Skipping test '${test}'!!"
      continue
  fi
  echo "# Running test script '${test}'"
  /usr/local/bin/bats --tap ${test}
done

docker rmi -f ${image}
