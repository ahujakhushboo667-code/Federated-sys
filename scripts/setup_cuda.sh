#!/bin/bash
set -e

echo "Setting up NVIDIA CUDA environment for FusionNet..."

# Install PyTorch with CUDA support (e.g. CUDA 12.1 or 11.8)
echo "Installing PyTorch for NVIDIA GPUs..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r ../requirements.txt

echo "Setup complete. Verifying environment..."
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
