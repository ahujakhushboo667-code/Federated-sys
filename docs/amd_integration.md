# AMD Hardware Integration

This document outlines how FusionNet leverages AMD hardware:
- **ROCm and PyTorch**: Native support for high-performance LoRA fine-tuning on MI300X and consumer GPUs.
- **bitsandbytes for 4-bit Quantization**: Utilizing the new ROCm 6.0 support in HuggingFace `bitsandbytes` to load models in `nf4` precision with double quantization.
- **Opacus for DP-SGD**: Wrapping the training loop with `opacus` to provide robust, mathematically sound Differential Privacy that is fully compatible with the ROCm backend.
- **RCCL for Secure Aggregation**: (Planned) Using RCCL backend in distributed PyTorch for efficient cross-device tensor averaging.
