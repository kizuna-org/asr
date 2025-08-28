#!/bin/bash

set -euo pipefail

trap 'echo "[kizuna-org/asr]Finished building docker images"' EXIT

pids=()
for dir in */ ; do
  if [ -f "${dir}Dockerfile" ]; then
    tag=$(basename "$dir")
    (
      cd "$dir"
      sudo docker build --network host -t "$tag" .
    ) &
    pids+=($!)
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
