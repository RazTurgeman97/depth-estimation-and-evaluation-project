#!/bin/bash

# Important Notes
# Docker hub EOL image cleanup script to run locally at this time with hub-tool.
# Download and extract contents https://github.com/docker/hub-tool/releases/download/v0.4.5/hub-tool-linux-amd64.tar.gz to /usr/local/bin for hub-tool setup
# Verify if the access token used to login with hub-tool has delete permission (https://hub.docker.com/settings/security)

DOCKER_TAGS=docker-file
EOL_TAGS=eol-file
DELETE_TAGS=delete-file

#cleanup
if [[ -f $DOCKER_TAGS && -f $EOL_TAGS && -f $DELETE_TAGS ]]; then
    rm $DOCKER_TAGS $EOL_TAGS $DELETE_TAGS
fi

if ! command -v hub-tool >/dev/null; then
  echo "==>hub-tool not found. This tool is required to launch api call to docker hub. Refer to description for setup instruction."
  exit 1
fi

echo "Proceeding to login..."
hub-tool login ${DOCKER_HUB_SVC_USER}
if [ $? -ne 0 ]; then
    echo "Login failed. Exiting!"
    exit 1
fi

#list all tags from docker hub
hub-tool tag ls --all nvidia/cuda | awk '{print $1}' > $DOCKER_TAGS

#compare the list from EOL with tags and create a new file for deletion
ls -ld dist/end-of-life/*/* | awk '{print $9}' | grep -v '^[[:blank:]]*$' | sort -V > $EOL_TAGS

if [[ -s $DOCKER_TAGS && -s $EOL_TAGS ]]; then
    while read -r eol_tag; do
        tag_version=`echo $eol_tag | awk -F/ '{print $3}'`
        tag_os=`echo $eol_tag | awk -F/ '{print $4}'`
        grep $tag_version $DOCKER_TAGS | grep $tag_os >> $DELETE_TAGS
    done < $EOL_TAGS
else
    echo "Docker tags or eol tags file doesn't exist or empty. Exiting!"
    exit 1
fi

#dryrun the deletion and ask for confirmation
if [[ -s $DELETE_TAGS ]]; then
    cat $DELETE_TAGS | awk '{print $1}'
    read -p "Please provide your confirmation to delete the listed tags. This action is irreversible (yes/no):" user_input
    if [[ $user_input == "yes" ]]; then
        while read -r delete_tag; do
            echo "Deleting $delete_tag"
            hub-tool tag rm $delete_tag --force
            if [ $? -ne 0 ]; then
                echo "==>hub-tool deletion failed for $delete_tag"
            fi
        done < $DELETE_TAGS
    fi
else
    echo "No tags to delete at this time. Exiting!"
    exit 0
fi

#cleanup
if [[ -f $DOCKER_TAGS && -f $EOL_TAGS && -f $DELETE_TAGS ]]; then
    rm $DOCKER_TAGS $EOL_TAGS $DELETE_TAGS
fi

