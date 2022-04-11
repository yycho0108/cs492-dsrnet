#!/usr/bin/env bash

set -ex

IMAGE_TAG='cs492'

# NOTE(ycho): Set context directory relative to this file.
CONTEXT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )/.." && pwd )"

## Build docker image.
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t "${IMAGE_TAG}" \
    -f "${CONTEXT_DIR}/docker/Dockerfile" \
    "${CONTEXT_DIR}"
