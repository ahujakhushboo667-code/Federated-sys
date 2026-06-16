from client import FusionNetClient

if __name__ == "__main__":
    client = FusionNetClient("config.yaml")
    client.fed_client.register_client()
    print("Starting local training round...")
    client.train()
    print("Exporting A updates...")
    updates = client.fed_client.export_A_update(round_num=1)
    print(f"Exported {len(updates)} A matrices.")
    print("Payload sample:", str(updates[0])[:100] + "...")
    print("FusionNet Client finished.")
