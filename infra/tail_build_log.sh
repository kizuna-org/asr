#!/bin/bash

set -euo pipefail

TARGET_HOST="edu-gpu"
TARGET_PORT=""
SSH_OPTS=""

if [[ "${1-}" == "-e" && "${2-}" == "mock" ]]; then
  TARGET_HOST="localhost"
  TARGET_PORT="50022"
  SSH_OPTS="-p ${TARGET_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
fi

ssh ${SSH_OPTS} "${TARGET_HOST}" "tail -f ~/frpclient/docker_pre_build.log"
