#!/bin/bash

set -euo pipefail

for tar in /home/ansible_user/for-cache/*.tar; do
  docker load -i "$tar"
done
