#!/usr/bin/env bash
# Install into the currently activated Python 3.12 virtual/Conda environment.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_env.sh [--gpu | --cpu]

  --gpu  CUDA 12.8 training stack, including FlashAttention (default).
         Requires Linux x86_64, a CUDA 12.8 toolkit, and a C++ compiler.
  --cpu  CPU PyTorch and test dependencies; no GPU training runtime.

Activate a fresh Python 3.12 environment first. Set PYTHON to choose its
interpreter and MAX_JOBS to limit FlashAttention build parallelism (default: 4).
EOF
}

MODE="${1:---gpu}"
if [[ "$MODE" == "--help" || "$MODE" == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -gt 1 || ( "$MODE" != "--gpu" && "$MODE" != "--cpu" ) ]]; then
    usage >&2
    exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
"$PYTHON_BIN" - <<'PY'
import os
import platform
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit("Use Python 3.12: python3.12 -m venv .venv && source .venv/bin/activate")
if sys.prefix == sys.base_prefix and not os.environ.get("CONDA_PREFIX"):
    raise SystemExit("Activate a virtual environment or Conda environment before installing.")
if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("This setup recipe targets Linux x86_64.")
PY

if [[ "$MODE" == "--gpu" ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "CUDA compiler nvcc was not found. Install/load CUDA Toolkit 12.8 first." >&2
        exit 1
    fi
    if [[ "$(nvcc --version)" != *"release 12.8"* ]]; then
        echo "This recipe requires CUDA Toolkit 12.8 to build FlashAttention against PyTorch cu128." >&2
        exit 1
    fi
fi

# vLLM 0.11.0 on Python 3.12 requires setuptools >=77.0.3,<80.
"$PYTHON_BIN" -m pip install --upgrade pip "setuptools==79.0.1" wheel
if [[ "$MODE" == "--cpu" ]]; then
    "$PYTHON_BIN" -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
    "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/requirements-dev.txt"
else
    "$PYTHON_BIN" -m pip install \
        "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
        --index-url https://download.pytorch.org/whl/cu128
    # OpenCV 4.12+ requires NumPy 2; the vendored runtime requires NumPy <2.
    "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/requirements-dev.txt" \
        "vllm==0.11.0" "liger-kernel==0.8.0" "opencv-python-headless==4.11.0.86"
    # Build against the PyTorch already installed above; vLLM supplies einops/ninja.
    MAX_JOBS="${MAX_JOBS:-4}" "$PYTHON_BIN" -m pip install \
        --no-build-isolation --no-deps "flash-attn==2.8.1"
fi

"$PYTHON_BIN" -m pip install --no-deps -e "$REPO_ROOT/verl"
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" - <<'PY'
import torch
import transformers
import verl

print(f"Installed: torch={torch.__version__}, transformers={transformers.__version__}, verl={verl.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY
echo "Setup complete. Run: python -m pytest -q tests"
