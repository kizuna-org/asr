#!/usr/bin/env bash

set -euo pipefail

# Local runner for asr-test using Docker and Docker Compose
# - Builds backend/frontend images locally
# - Starts services via docker compose
# - Prints access URLs

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.yml"
COMPOSE_GPU_FILE="${PROJECT_DIR}/docker-compose.gpu.yml"

# (compose build へ移行したため未使用)
# IMAGE_BACKEND="asr-app"
# IMAGE_FRONTEND="asr-frontend"

PORT_FRONTEND=58080
PORT_BACKEND=58081

DO_BUILD=1
DO_PULL=0
DO_DOWN=1
USE_PROXY=0
USE_GPU=0

print_usage() {
    echo "Usage: $0 [--no-build] [--pull] [--no-down] [--use-proxy] [--gpu] [--help]"
    echo "  --no-build  Skip docker build steps"
    echo "  --pull      Run 'docker compose pull' before up"
    echo "  --no-down   Do not run 'docker compose down' before up"
    echo "  --use-proxy Pass HTTP(S)_PROXY/NO_PROXY as build-args"
    echo "  --gpu       Force GPU mode (use docker-compose.gpu.yml)"
    echo "  --help      Show this help"
}

for arg in "$@"; do
    case "$arg" in
        --no-build)
            DO_BUILD=0
            ;;
        --pull)
            DO_PULL=1
            ;;
        --no-down)
            DO_DOWN=0
            ;;
        --use-proxy)
            USE_PROXY=1
            ;;
        --gpu)
            USE_GPU=1
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            print_usage
            exit 1
            ;;
    esac
done

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "❌ Required command not found: $1" >&2
        exit 1
    fi
}

echo "🔎 Checking prerequisites..."
require_cmd docker


if ! docker compose version >/dev/null 2>&1; then
    echo "❌ 'docker compose' is not available. Please install Docker Desktop (v2.20+) or Compose V2." >&2
    exit 1
fi

if [ ! -f "${COMPOSE_FILE}" ]; then
    echo "❌ docker-compose.yml not found at ${COMPOSE_FILE}" >&2
    exit 1
fi

# GPU環境の自動検出
if [ "${USE_GPU}" -eq 0 ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "🔍 NVIDIA GPU detected, enabling GPU mode"
        USE_GPU=1
    elif docker info 2>/dev/null | grep -q "nvidia"; then
        echo "🔍 NVIDIA Container Runtime detected, enabling GPU mode"
        USE_GPU=1
    else
        echo "💻 No GPU detected, running in CPU mode"
    fi
fi

# Composeファイルの選択
if [ "${USE_GPU}" -eq 1 ]; then
    if [ ! -f "${COMPOSE_GPU_FILE}" ]; then
        echo "❌ docker-compose.gpu.yml not found at ${COMPOSE_GPU_FILE}" >&2
        exit 1
    fi
    COMPOSE_FILES="-f ${COMPOSE_FILE} -f ${COMPOSE_GPU_FILE}"
    echo "🚀 Using GPU-enabled compose configuration"
else
    COMPOSE_FILES="-f ${COMPOSE_FILE}"
    echo "🚀 Using CPU-only compose configuration"
fi

if [ "${DO_DOWN}" -eq 1 ]; then
    echo "🛑 Stopping existing containers (if any)..."
    docker compose ${COMPOSE_FILES} down --remove-orphans || true
fi

BUILD_ARGS=()
if [ "${USE_PROXY}" -eq 1 ]; then
    if [ -n "${HTTP_PROXY:-}" ]; then BUILD_ARGS+=(--build-arg "HTTP_PROXY=${HTTP_PROXY}"); fi
    if [ -n "${HTTPS_PROXY:-}" ]; then BUILD_ARGS+=(--build-arg "HTTPS_PROXY=${HTTPS_PROXY}"); fi
    if [ -n "${NO_PROXY:-}" ]; then BUILD_ARGS+=(--build-arg "NO_PROXY=${NO_PROXY}"); fi
fi

if [ "${DO_BUILD}" -eq 1 ]; then
    echo "🔨 Building images via docker compose (ensures compose image names are rebuilt)"
    if ((${#BUILD_ARGS[@]:-0})); then
        docker compose ${COMPOSE_FILES} build "${BUILD_ARGS[@]}"
    else
        docker compose ${COMPOSE_FILES} build
    fi
else
    echo "⏭️  Skipping image build as requested"
fi

if [ "${DO_PULL}" -eq 1 ]; then
    echo "📥 Pulling service images as requested..."
    docker compose ${COMPOSE_FILES} pull
fi

echo "🚀 Starting services with Docker Compose..."
docker compose ${COMPOSE_FILES} up -d

echo "⏳ Waiting for containers to become healthy (best-effort)..."
sleep 2

echo "📋 Current service status:"
docker compose ${COMPOSE_FILES} ps

echo ""
echo "🌐 Frontend:  http://localhost:${PORT_FRONTEND}"
echo "🔗 Backend:   http://localhost:${PORT_BACKEND}/docs"
echo ""
echo "💡 Notes:"
echo "- Default build does NOT pass host proxy. Use --use-proxy to forward HTTP(S)_PROXY/NO_PROXY."
echo "- GPU mode is auto-detected. Use --gpu to force GPU mode or run without GPU detection."
echo "- Backend uses CUDA base image. It can run without GPU, but GPU features will be unavailable."
echo ""
echo "✅ Done. Use 'docker compose ${COMPOSE_FILES} logs -f' to tail logs."


