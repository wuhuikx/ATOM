#!/usr/bin/env bash
# Build AtomRL Docker images.
#
# Usage:
#   bash atomrl/docker/build.sh              # build ATOM base + AtomRL image
#   bash atomrl/docker/build.sh --dev        # same image, tagged as atomrl:dev
#   bash atomrl/docker/build.sh --base IMG   # use an existing Atom base image
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOMRL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ATOMRL_ROOT}/.." && pwd)"

DEV_MODE=false
BASE_IMAGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev) DEV_MODE=true; shift ;;
        --base) BASE_IMAGE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "${REPO_ROOT}"

if [ ! -d "atomrl/lumen" ]; then
    echo "ERROR: unified Lumen source not found at atomrl/lumen"
    exit 1
fi
if [ ! -d "atom" ]; then
    echo "ERROR: run this script from an ATOM checkout containing atom/"
    exit 1
fi

DOCKERFILE="atomrl/docker/Dockerfile.dev"
if [ "${DEV_MODE}" = true ]; then
    TAG="atomrl:dev"
else
    TAG="atomrl:latest"
fi

BUILD_ARGS=""
if [ -n "${BASE_IMAGE}" ]; then
    BUILD_ARGS="--build-arg BASE_IMAGE=${BASE_IMAGE}"
else
    echo "Building ATOM base image from docker/Dockerfile..."
    docker build \
        --pull \
        --no-cache \
        -f "docker/Dockerfile" \
        -t "rocm/atom-dev:latest" \
        .
    BUILD_ARGS="--build-arg BASE_IMAGE=rocm/atom-dev:latest"
fi

echo "Building ${TAG} from ${DOCKERFILE}..."
docker build \
    --pull \
    --no-cache \
    -f "${DOCKERFILE}" \
    ${BUILD_ARGS} \
    -t "${TAG}" \
    .

echo "Done. Run with:"
echo "  docker run --rm -it --device /dev/kfd --device /dev/dri \\"
echo "    -v /dev/shm:/dev/shm --shm-size=256g \\"
echo "    --network=host ${TAG}"
