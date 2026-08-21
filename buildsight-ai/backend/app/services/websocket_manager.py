"""BuildSight AI — WebSocket Connection Manager"""

from fastapi import WebSocket
from typing import Optional
import json
import logging
import asyncio

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts analytics updates."""

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self._connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self._connections)}")

    async def broadcast_json(self, data: dict):
        """Broadcast JSON data to all connected clients."""
        if not self._connections:
            return

        message = json.dumps(data, default=str)
        disconnected = []

        async with self._lock:
            for ws in self._connections:
                try:
                    await ws.send_text(message)
                except Exception:
                    disconnected.append(ws)

        # Clean up disconnected clients
        for ws in disconnected:
            async with self._lock:
                if ws in self._connections:
                    self._connections.remove(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Global instance
ws_manager = WebSocketManager()
