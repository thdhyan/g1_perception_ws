#!/usr/bin/env bash
# ==============================================================================
# Script to create and configure the uv Python environment for g1_perception_ws
# ==============================================================================
set -e

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WS_DIR"

echo "=== [1/4] Checking for uv package manager ==="
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv via curl..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "=== [2/4] Creating virtual environment (.venv) ==="
# System site packages allows ROS2 jazzy python modules to be accessible
uv venv --python 3.12 --system-site-packages .venv

echo "=== [3/4] Installing PyTorch (cu121), SpConv, ONNX, and PointPillars dependencies ==="
source .venv/bin/activate

uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt

echo "=== [4/4] Verifying CUDA and PyTorch ==="
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | Devices: {torch.cuda.device_count()}')"
python3 -c "import onnxruntime as ort; print(f'ONNX Runtime {ort.__version__} | Providers: {ort.get_available_providers()}')"

echo "=============================================================================="
echo " UV Environment created successfully in $WS_DIR/.venv"
echo " Activate with: source setup_g1_env.sh"
echo "=============================================================================="
