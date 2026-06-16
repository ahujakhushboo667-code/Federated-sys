import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import FusionNetClient

def simulate_coordinator_aggregation(client_updates):
    print("\n[Coordinator] Receiving client updates...")
    aggregated_a = client_updates[0]
    print("[Coordinator] Aggregation complete. Broadcasting new A matrices...")
    return aggregated_a

if __name__ == "__main__":
    print("--- FusionNet Example Federated Round ---")
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    client = FusionNetClient(config_path)
    client.fed_client.register_client()
    
    print("\n[Client] Starting local training epoch...")
    # Instead of a full train which takes time, we just save to trigger logic
    client.fed_client.save_local_adapter()
    print("[Client] Extracting A matrix updates...")
    
    updates = client.fed_client.export_A_update(round_num=1)
    
    global_a_update = simulate_coordinator_aggregation([updates])
    
    print("\n[Client] Receiving global A matrices...")
    client.fed_client.receive_global_A(global_a_update)
    
    print("\nFederated round cycle complete!")
