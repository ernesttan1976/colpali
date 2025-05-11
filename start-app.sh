#!/bin/bash
# Startup script for the ColPali Docker application

# Check environment
echo "=== System Information ==="
uname -a
nvidia-smi || echo "NVIDIA drivers not found or not accessible"

# Verify CUDA
echo "=== Running CUDA Verification ==="
python verify_cuda.py

# Handle flash-attn installation
echo "=== Checking Flash Attention Requirement ==="
if grep -q "install_fa2" app.py; then
  echo "Flash Attention is used in the app, but we'll let the app handle its installation if needed"
  echo "If the app tries to install flash-attn automatically, it might take several minutes"
  echo "You can check app.py if it actually needs flash-attn or if it can be disabled"
fi

# Verify torch version
echo "=== PyTorch Configuration ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Create necessary directories
mkdir -p ./data/embeddings_db ./models

# Launch application
echo "=== Starting ColPali Application ==="
python app.py