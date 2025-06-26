#!/bin/bash
# docker_build.sh: 指定したDockerfileの全FROMイメージを事前にpullしてからdocker buildを実行するスクリプト
# 使い方: ./docker_build.sh -f <Dockerfileパス> [ビルドコンテキスト] [--no-sudo]

set -e

usage() {
  echo "Usage: $0 -f <Dockerfile> [build context] [docker build args...] [--no-sudo]"
  exit 1
}

DOCKERFILE=""
CONTEXT="."
BUILD_ARGS=()
DOCKER_CMD="sudo docker"

# 引数パース
while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--file)
      DOCKERFILE="$2"
      shift 2
      ;;
    --no-sudo)
      DOCKER_CMD="docker"
      shift
      ;;
    *)
      if [[ -z "$CONTEXT_SET" ]]; then
        CONTEXT_SET=1
        CONTEXT="$1"
        shift
      else
        BUILD_ARGS+=("$1")
        shift
      fi
      ;;
  esac
done

if [[ -z "$DOCKERFILE" ]]; then
  usage
fi

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "Dockerfile not found: $DOCKERFILE"
  exit 1
fi

# FROMイメージを全て抽出してpull
IMAGES=$(grep -E '^FROM ' "$DOCKERFILE" | awk '{print $2}' | sort | uniq)
echo "🔍 Pulling base images used in $DOCKERFILE..."
for IMAGE in $IMAGES; do
  echo "$DOCKER_CMD pull $IMAGE"
  $DOCKER_CMD pull "$IMAGE"
done
echo "✅ All base images pulled."

echo "🚀 Building image with $DOCKER_CMD build..."
$DOCKER_CMD build -f "$DOCKERFILE" "$CONTEXT" "${BUILD_ARGS[@]}"
echo "✅ Build completed." 
