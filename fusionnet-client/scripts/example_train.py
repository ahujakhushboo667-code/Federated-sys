import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import FusionNetClient

if __name__ == "__main__":
    print("--- FusionNet Example Local Training ---")
    # Make sure we use the config from the project root
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    client = FusionNetClient(config_path)
    
    print("\nStarting local training...")
    client.train()
    
    print("\nTraining completed. Adapters saved to checkpoints/")
