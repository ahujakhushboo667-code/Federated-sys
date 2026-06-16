#!/bin/bash
set -e

echo "Setting up NVIDIA CUDA environment for FusionNet..."

# Requires driver >= 560 for CUDA 12.8 (Blackwell RTX 50xx, Ada RTX 40xx, Ampere RTX 30xx).
# Install PyTorch cu128 — the first stable wheel set supporting Blackwell (sm_100).
echo "Installing PyTorch cu128..."
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies (torch is excluded from requirements.txt)
pip install -r ../requirements.txt

echo ""
echo "Setup complete. Verifying environment..."
python -c "
import os, sys
if sys.platform == 'win32':
    os.add_dll_directory(r'C:\Windows\System32')
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU     : {p.name} ({p.total_memory / 1024**3:.1f} GB)')
"
