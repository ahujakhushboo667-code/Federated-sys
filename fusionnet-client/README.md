# FusionNet Local Client

This is the local client component of FusionNet, a privacy-preserving federated learning system optimized for AMD hardware. 

## Architecture Requirements
- **Frozen Base Model**: Llama 3-8B in 4-bit NF4 quantization. Base weights remain frozen. No full fine-tuning allowed.
- **AFLoRA Adapter**: Splits weight updates into `ΔW = A × Λ × B`, keeping `B` and `Λ` strictly on-device for personalization. Only `A` is sent to the coordinator.
- **Hardware-Aware**: Automatically detects the environment (MI300X, RX 7900 XTX, Steam Deck, CPU) and scales adapter rank and precision accordingly.
- **Differential Privacy**: Includes Opacus-powered DP-SGD with a resilient custom fallback for 4-bit quantized modules.
- **Communication Protocol**: High-performance Base64 serialization of `A` matrices for JSON communication.

## Quickstart

```bash
cd fusionnet-client
pip install -r requirements.txt
```

### Run Local Node
```bash
python main.py
```

### Run Examples
```bash
python scripts/example_train.py
python scripts/example_federated_round.py
```

## Structure
- `models/`: HuggingFace Llama 3 4-bit loader.
- `aflora/`: AFLoRA module and HuggingFace injection utilities.
- `federation/`: Client networking, base64 serialization, and Differential Privacy logic.
- `training/`: Local training loop.
- `datasets/`: Dataset loading utilities (Banking77, SST-2, IMDB, AG News).
