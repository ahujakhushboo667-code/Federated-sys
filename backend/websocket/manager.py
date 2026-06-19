import asyncio
from fastapi import WebSocket
from typing import Dict, List
import json

class WebSocketManager:
    def __init__(self):
        # Dictionary to hold active connections per channel
        # Channels: 'devices', 'rounds', 'metrics', 'events', 'all'
        self.active_connections: Dict[str, List[WebSocket]] = {
            "devices": [],
            "rounds": [],
            "metrics": [],
            "events": [],
            "all": []
        }

    async def connect(self, websocket: WebSocket, channel: str) -> bool:
        total_connections = sum(len(conns) for conns in self.active_connections.values())
        if total_connections >= 100:
            await websocket.accept()
            await websocket.close(code=1008, reason="Connection limit exceeded")
            return False

        await websocket.accept()
        if channel in self.active_connections:
            self.active_connections[channel].append(websocket)
        else:
            self.active_connections[channel] = [websocket]
        return True

    def disconnect(self, websocket: WebSocket, channel: str):
        if channel in self.active_connections and websocket in self.active_connections[channel]:
            self.active_connections[channel].remove(websocket)

    async def broadcast(self, channel: str, message: dict):
        targets = []
        if channel in self.active_connections:
            targets.extend(self.active_connections[channel])
        
        if channel != "all" and "all" in self.active_connections:
            targets.extend(self.active_connections["all"])

        # Deduplicate targets
        unique_targets = list(set(targets))

        async def _send_safe(ws: WebSocket):
            try:
                await ws.send_json(message)
            except Exception:
                pass

        if unique_targets:
            await asyncio.gather(*(_send_safe(ws) for ws in unique_targets))

manager = WebSocketManager()
