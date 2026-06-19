import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.websocket.manager import manager
import logging

router = APIRouter(prefix="/ws", tags=["websocket"])
logger = logging.getLogger(__name__)

@router.websocket("/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    connected = await manager.connect(websocket, channel)
    if not connected:
        return

    logger.info(f"WebSocket connected to channel: {channel}")
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                logger.warning(f"WebSocket idle timeout (30s) on channel: {channel}. Disconnecting client.")
                break
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected by client from channel: {channel}")
    finally:
        manager.disconnect(websocket, channel)
        logger.info(f"WebSocket cleaned up from channel: {channel}")
