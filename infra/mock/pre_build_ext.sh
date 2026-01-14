#!/bin/bash

set -euo pipefail

trap 'echo "[kizuna-org/asr]Finished building docker images"' EXIT

mkdir -p infra/mock/dind-host/for-cache/
pids=()
for dir in infra/frpc/* ; do
  if [ -f "${dir}Dockerfile" ]; then
    tag=$(basename "$dir")
    (
      cd "$dir"
      docker build -t "$tag" .
      docker save "$tag" > infra/mock/dind-host/for-cache/"$tag.tar"
    ) &
    pids+=($!)
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
