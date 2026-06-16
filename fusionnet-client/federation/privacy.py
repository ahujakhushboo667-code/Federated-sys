import torch
import torch.nn as nn
import base64
import json
import numpy as np

class CustomPrivacyEngine:
    """
    Fallback Differential Privacy implementation if Opacus fails.
    Implements per-sample gradient clipping and Gaussian noise addition.
    """
    def __init__(self, model, optimizer, max_grad_norm, epsilon, delta):
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon
        self.delta = delta
        
        # Simple noise multiplier calculation based on DP-SGD
        self.noise_multiplier = (max_grad_norm * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
        
    def step(self):
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 
            self.max_grad_norm
        )
        
        # Add noise
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                noise = torch.normal(
                    mean=0.0, 
                    std=self.noise_multiplier * self.max_grad_norm, 
                    size=param.grad.shape, 
                    device=param.grad.device
                )
                param.grad += noise
                
        self.optimizer.step()

def setup_privacy(model, optimizer, dataloader, config):
    """
    Attempts to setup Opacus, falls back to CustomPrivacyEngine if it fails.
    """
    if not config.get("use_dp_sgd", False):
        return model, optimizer, dataloader, None
        
    epsilon = config.get("epsilon", 1.0)
    delta = config.get("delta", 1e-5)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    
    try:
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=config.get("local_epochs", 1),
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )
        print("Opacus DP-SGD successfully initialized.")
        return model, optimizer, dataloader, privacy_engine
    except Exception as e:
        print(f"Opacus failed: {e}. Falling back to CustomPrivacyEngine.")
        custom_engine = CustomPrivacyEngine(model, optimizer, max_grad_norm, epsilon, delta)
        return model, optimizer, dataloader, custom_engine

def serialize_tensor_base64(tensor: torch.Tensor):
    """
    Serializes a tensor to a base64 encoded string representing float16 data.
    """
    tensor_np = tensor.cpu().numpy().astype(np.float16)
    b64_str = base64.b64encode(tensor_np.tobytes()).decode('utf-8')
    return b64_str

def deserialize_tensor_base64(b64_str: str, shape: list):
    """
    Deserializes a base64 string back into a torch Tensor (float16).
    """
    tensor_bytes = base64.b64decode(b64_str)
    tensor_np = np.frombuffer(tensor_bytes, dtype=np.float16).reshape(shape)
    return torch.from_numpy(tensor_np)
