#!/bin/bash

set -euo pipefail

for dir in */ ; do
  if [ -f "${dir}Dockerfile" ]; then
    tag=$(basename "$dir")
    (
      cd "$dir"
      sudo docker build -t "$tag" .
    )
  fi
done
