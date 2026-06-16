import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import psutil

class DeviceDetector:
    @staticmethod
    def detect_hardware():
        # Heuristic device profiling
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if "mi300x" in gpu_name or vram_gb >= 100:
                return "MI300X", "cuda" # ROCm appears as cuda in PyTorch
            elif "7900" in gpu_name or vram_gb >= 20:
                return "RX_7900_XTX", "cuda"
            else:
                return "Steam_Deck", "cuda" # Fallback for smaller GPUs
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Steam_Deck", "mps" # Equivalent memory profile for Apple Silicon
        else:
            return "CPU_only", "cpu"

def load_llama(model_name: str, quantization_type: str = "nf4"):
    device_profile, device_type = DeviceDetector.detect_hardware()
    print(f"Detected hardware profile: {device_profile} on {device_type}")
    
    quantization_config = None
    if device_type == "cuda":
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=quantization_type,
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            print("bitsandbytes not installed, falling back to unquantized")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if device_type == "cuda" else None,
        torch_dtype=torch.float16 if device_type != "cpu" else torch.float32,
    )
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
        
    return model, tokenizer, device_profile
