#!/usr/bin/env bash

set -ex

IMAGE_TAG='cs492-dsr'

# Figure out repository root.
SCRIPT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# Launch simple docker container with
# * Network enabled (passthrough to host)
# * Privileged permissions
# * All GPU devices visible
# * Current working git repository mounted at /root
docker run -it --rm \
    --mount type=bind,source=${REPO_ROOT},target="/root/$(basename ${REPO_ROOT})" \
    --mount type=bind,source=/tmp/,target="/tmp/host/" \
    --mount type=bind,source=/media/ssd/datasets/DSR,target="/media/ssd/datasets/DSR" \
    --network host \
    --privileged \
    --gpus all \
    "$@" \
    "${IMAGE_TAG}"
