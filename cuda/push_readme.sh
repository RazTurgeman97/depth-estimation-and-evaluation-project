#!/bin/bash

set -euo pipefail

# DOCKER HUB
#
CMD="curl -s -H 'Content-Type:application/json' -d \"{\\\"username\\\":\\\"${DOCKER_HUB_SVC_USER}\\\",\\\"password\\\":\\\"${DOCKER_HUB_SVC_PASS}\\\"}\" https://hub.docker.com/v2/users/login"
echo "COMMAND: ${CMD}"

TOKEN=$(echo -e "$CMD" | source /dev/stdin | jq -r .token)
echo "TOKEN: ${TOKEN}"

jq -n --arg msg "$(<doc/README.md)" '{"full_description": $msg }' | \
    curl -sv -L -w "%{http_code}" \
       https://hub.docker.com/v2/repositories/nvidia/cuda/ \
       -d @- -X PATCH \
       -H "Content-Type: application/json" \
       -H "Authorization: Bearer ${TOKEN}"

# JSON_DATA="{\\\"full_description\\\": \\\"$(cat doc/README.md)\\\"}"
# CMD2="curl -vs -x PATCH -H 'Authorization:Bearer ${TOKEN}' -H 'Content-Type:application/json' -d '${JSON_DATA}' https://hub.docker.com/v2/repositories/nvidia/cuda/"
# echo "COMMAND: ${CMD2}"
# # bash quoting just kills me
# echo -e "$CMD2" | source /dev/stdin


#
# NGC
#
# ngc registry image update nvidia/cuda  --overview doc/README.md
