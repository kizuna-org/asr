#!/usr/bin/env bash

set -euo pipefail

# Local runner for asr-test using Docker and Docker Compose
# - Builds backend/frontend images locally
# - Starts services via docker compose
# - Prints access URLs

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.yml"

IMAGE_BACKEND="asr-app"
IMAGE_FRONTEND="asr-frontend"

PORT_FRONTEND=58080
PORT_BACKEND=58081

DO_BUILD=1
DO_PULL=0
DO_DOWN=1
USE_PROXY=0

print_usage() {
    echo "Usage: $0 [--no-build] [--pull] [--no-down] [--use-proxy] [--help]"
    echo "  --no-build  Skip docker build steps"
    echo "  --pull      Run 'docker compose pull' before up"
    echo "  --no-down   Do not run 'docker compose down' before up"
    echo "  --use-proxy Pass HTTP(S)_PROXY/NO_PROXY as build-args"
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

if [ "${DO_DOWN}" -eq 1 ]; then
    echo "🛑 Stopping existing containers (if any)..."
    docker compose -f "${COMPOSE_FILE}" down --remove-orphans || true
fi

BUILD_ARGS=()
if [ "${USE_PROXY}" -eq 1 ]; then
    if [ -n "${HTTP_PROXY:-}" ]; then BUILD_ARGS+=(--build-arg "HTTP_PROXY=${HTTP_PROXY}"); fi
    if [ -n "${HTTPS_PROXY:-}" ]; then BUILD_ARGS+=(--build-arg "HTTPS_PROXY=${HTTPS_PROXY}"); fi
    if [ -n "${NO_PROXY:-}" ]; then BUILD_ARGS+=(--build-arg "NO_PROXY=${NO_PROXY}"); fi
fi

if [ "${DO_BUILD}" -eq 1 ]; then
    echo "🔨 Building backend image: ${IMAGE_BACKEND}"
    if ((${#BUILD_ARGS[@]:-0})); then
        docker build "${PROJECT_DIR}" -f "${PROJECT_DIR}/backend/Dockerfile" -t "${IMAGE_BACKEND}" "${BUILD_ARGS[@]}"
    else
        docker build "${PROJECT_DIR}" -f "${PROJECT_DIR}/backend/Dockerfile" -t "${IMAGE_BACKEND}"
    fi

    echo "🔨 Building frontend image: ${IMAGE_FRONTEND}"
    if ((${#BUILD_ARGS[@]:-0})); then
        docker build "${PROJECT_DIR}" -f "${PROJECT_DIR}/frontend/Dockerfile" -t "${IMAGE_FRONTEND}" "${BUILD_ARGS[@]}"
    else
        docker build "${PROJECT_DIR}" -f "${PROJECT_DIR}/frontend/Dockerfile" -t "${IMAGE_FRONTEND}"
    fi
else
    echo "⏭️  Skipping image build as requested"
fi

if [ "${DO_PULL}" -eq 1 ]; then
    echo "📥 Pulling service images as requested..."
    docker compose -f "${COMPOSE_FILE}" pull
fi

echo "🚀 Starting services with Docker Compose..."
docker compose -f "${COMPOSE_FILE}" up -d

echo "⏳ Waiting for containers to become healthy (best-effort)..."
sleep 2

echo "📋 Current service status:"
docker compose -f "${COMPOSE_FILE}" ps

echo ""
echo "🌐 Frontend:  http://localhost:${PORT_FRONTEND}"
echo "🔗 Backend:   http://localhost:${PORT_BACKEND}/docs"
echo ""
echo "💡 Notes:"
echo "- Default build does NOT pass host proxy. Use --use-proxy to forward HTTP(S)_PROXY/NO_PROXY."
echo "- Backend uses CUDA base image. It can run without GPU, but GPU features will be unavailable."
echo "- If you have NVIDIA runtime installed, Compose may utilize it automatically."
echo ""
echo "✅ Done. Use 'docker compose -f ${COMPOSE_FILE} logs -f' to tail logs."


