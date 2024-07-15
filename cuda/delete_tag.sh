#!/bin/bash

set -euo pipefail


# DOCKER HUB
#
#
CMD="curl -s -H 'Content-Type:application/json' -d \"{\\\"username\\\":\\\"${DOCKER_HUB_SVC_USER}\\\",\\\"password\\\":\\\"${DOCKER_HUB_SVC_PASS}\\\"}\" https://hub.docker.com/v2/users/login"
echo "COMMAND: ${CMD}"

TOKEN=$(echo -e "$CMD" | source /dev/stdin | jq -r .token)
echo "TOKEN: ${TOKEN}"

# TAG="nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu16.04"
jq -n "{\"dry_run\": false, \"manifests\":[{\"repository\":\"cuda\", \"digest\":\"sha256:7723438fcb16fcab57718e7642ca904fc25573df5e4ac6bc6be04317da21c8b2\"}]}" | \
    curl -sv -L -w "%{http_code}" \
       https://hub.docker.com/v2/namespaces/nvidia/delete-images \
       -d @- -X POST \
       -H "Content-Type: application/json" \
       -H "Authorization: Bearer ${TOKEN}"

#
# NGC
#
# Not allowed.
