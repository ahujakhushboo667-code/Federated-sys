#!/bin/bash
set -e

echo "Setting up AMD ROCm environment for FusionNet..."

# Requires ROCm 6.1+ on Linux (Ubuntu 22.04 recommended).
# Install PyTorch for ROCm 6.1.
echo "Installing PyTorch for ROCm 6.1..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1

# Install remaining dependencies (torch is excluded from requirements.txt)
pip install -r ../requirements.txt

echo ""
echo "Setup complete. Verifying environment..."
python -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'ROCm    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU     : {p.name} ({p.total_memory / 1024**3:.1f} GB)')
"
