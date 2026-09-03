#!/bin/bash
# Build all images and push to GHCR.
# Usage:
#   ./docker/build_and_push.sh              # x86_64, tag: latest
#   PLATFORM=linux/arm64 TAG=orin ./docker/build_and_push.sh   # Jetson Orin
#   PLATFORM=linux/arm64 TAG=xavier \
#     JETPACK=5 ./docker/build_and_push.sh  # Jetson Xavier (JetPack 5)

set -e
cd "$(dirname "$0")/.."

REGISTRY="${REGISTRY:-thdhyan}"
TAG="${TAG:-latest}"
PLATFORM="${PLATFORM:-linux/amd64}"
JETPACK="${JETPACK:-6}"

# Jetson build args
if [[ "$PLATFORM" == "linux/arm64" ]]; then
    if [[ "$JETPACK" == "5" ]]; then
        JETSON_BASE="dustynv/ros:humble-ros-base-l4t-r35.4.1"
        TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
    else
        JETSON_BASE="dustynv/ros:humble-ros-base-l4t-r36.2.0"
        TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+6ddf5cf85e.nv24.04-cp310-cp310-linux_aarch64.whl"
    fi
    EXTRA_ARGS="--build-arg BASE=${JETSON_BASE} --build-arg TORCH_URL=${TORCH_URL} --build-arg BUILD_PYG=1"
else
    EXTRA_ARGS=""
fi

BUILD="docker buildx build --platform=${PLATFORM} ${EXTRA_ARGS} --push"

echo "=== Building for ${PLATFORM} tag=${TAG} ==="

$BUILD -f docker/Dockerfile.sensors  -t ${REGISTRY}/g1-sensors:${TAG}   .
$BUILD -f docker/Dockerfile.detection -t ${REGISTRY}/g1-detection:${TAG} .
$BUILD -f docker/Dockerfile.hmr      -t ${REGISTRY}/g1-hmr:${TAG}        .
$BUILD -f docker/Dockerfile.reid     -t ${REGISTRY}/g1-reid:${TAG}        .
$BUILD -f docker/Dockerfile.control  -t ${REGISTRY}/g1-control:${TAG}     .

echo "=== All images pushed to ${REGISTRY} with tag ${TAG} ==="
echo ""
echo "Pull on target device:"
echo "  docker compose -f docker/compose.yaml pull"
echo "  docker compose -f docker/compose.yaml up"
