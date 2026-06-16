import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Ensure fusionnet is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fusionnet.models.llama_loader import load_llama_4bit
from fusionnet.core.fl_coordinator import FLCoordinator
from fusionnet.core.aggregator import fed_avg

def create_mock_dataloader(batch_size=2, num_samples=16, seq_len=32):
    """Creates a mock dataloader with random tokens for testing."""
    # Assuming vocab size of 32000 (typical for Llama)
    input_ids = torch.randint(0, 32000, (num_samples, seq_len))
    labels = input_ids.clone()
    
    dataset = TensorDataset(input_ids, labels)
    
    # Custom collate function to output dicts as expected by LoRATrainer
    def collate_fn(batch):
        b_input_ids = torch.stack([item[0] for item in batch])
        b_labels = torch.stack([item[1] for item in batch])
        return {
            "input_ids": b_input_ids,
            "labels": b_labels
        }
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def main():
    print("=== Testing FusionNet Local Training Pipeline ===")
    
    # 1. Load Model dynamically based on hardware
    from fusionnet.models.model_selector import select_model_for_hardware
    selection = select_model_for_hardware()
    model_id = selection["model_id"]
    print(f"Using model: {model_id} ({selection['reason']})")
    
    try:
        model, tokenizer = load_llama_4bit(model_id, lora_rank=4)
    except Exception as e:
        print(f"Failed to load 4-bit model (this is expected if bitsandbytes/ROCm is not fully configured). Error: {e}")
        print("Falling back to a dummy Linear layer test to verify DP and orchestration logic...")
        
        # Fallback to a tiny dummy model for testing logic without GPU/bitsandbytes
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(32, 32)
            def forward(self, input_ids, attention_mask=None, labels=None):
                # Dummy forward pass
                out = self.fc(input_ids.float())
                loss = torch.nn.functional.mse_loss(out, labels.float()) if labels is not None else None
                from collections import namedtuple
                Output = namedtuple('Output', ['loss'])
                return Output(loss=loss)
                
        model = DummyModel()
        # Mock PEFT wrapping logic
        for param in model.parameters():
            param.requires_grad = True

    # 2. Setup Coordinator
    coordinator = FLCoordinator(model)
    
    # 3. Create Mock Data
    dataloader = create_mock_dataloader()
    
    # 4. Simulate a local FL Round
    print("\n--- Simulating Round 1 ---")
    # No global weights for round 1
    updated_weights_round1, metrics_round1 = coordinator.start_round(
        global_weights=None,
        local_dataloader=dataloader,
        epochs=1,
        dp_epsilon=1.5,
        dp_delta=1e-5
    )
    
    print("\nMetrics Round 1:", metrics_round1)
    
    # 5. Test Aggregation (Simulate 2 clients)
    print("\n--- Testing FedAvg Aggregation ---")
    # Simulate a second client by slightly modifying the weights
    client2_weights = {}
    for k, v in updated_weights_round1.items():
        client2_weights[k] = v + (torch.randn_like(v) * 0.01)
        
    client_weights = [updated_weights_round1, client2_weights]
    client_sizes = [16, 32] # Client 2 has twice the data
    
    aggregated_weights = fed_avg(client_weights, client_sizes)
    
    print("Aggregation successful!")
    print("Sample averaged weight tensor shape:", next(iter(aggregated_weights.values())).shape)

if __name__ == "__main__":
    main()
