import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def load_llama_4bit(model_id_or_path, lora_rank=8, lora_alpha=16, lora_dropout=0.05):
    """
    Loads a Llama model in 4-bit precision and injects a LoRA adapter.
    
    Args:
        model_id_or_path (str): HuggingFace model ID or local path.
        lora_rank (int): Rank of the LoRA matrices.
        lora_alpha (int): Alpha parameter for LoRA scaling.
        lora_dropout (float): Dropout probability for LoRA layers.
        
    Returns:
        model (PeftModel): The 4-bit quantized model with LoRA adapters.
        tokenizer (PreTrainedTokenizer): The associated tokenizer.
    """
    print(f"Loading tokenizer from {model_id_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading 4-bit model from {model_id_or_path}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, # or bfloat16 for newer hardware
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        quantization_config=quantization_config,
        device_map="auto", # Maps to available ROCm GPU automatically
    )

    # Prepare model for k-bit training (freezes base weights, sets requires_grad)
    model = prepare_model_for_kbit_training(model)

    print(f"Injecting LoRA adapter (r={lora_rank}, alpha={lora_alpha})...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    return peft_model, tokenizer
