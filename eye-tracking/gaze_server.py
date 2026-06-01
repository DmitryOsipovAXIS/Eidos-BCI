"""WebSocket server that broadcasts gaze events to connected EIDOS clients."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class GazeWebSocketServer:
    """Thread-safe gaze broadcaster with a background asyncio event loop."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._clients: Set[WebSocketServerProtocol] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._stop_event: Optional[asyncio.Event] = None

    async def _handler(self, websocket: WebSocketServerProtocol) -> None:
        self._clients.add(websocket)
        logger.info("Client connected (%d total)", len(self._clients))
        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected (%d total)", len(self._clients))

    async def _broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        message = json.dumps(payload)
        for client in list(self._clients):
            try:
                await client.send(message)
            except websockets.ConnectionClosed:
                self._clients.discard(client)

    def send_gaze(self, gaze: str) -> None:
        """Enqueue a gaze message from the tracking thread (call only on state change)."""
        if self._loop is None or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast({"gaze": gaze}), self._loop)

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        async with websockets.serve(self._handler, self.host, self.port):
            logger.info("Gaze WebSocket server at ws://%s:%d", self.host, self.port)
            self._ready.set()
            await self._stop_event.wait()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    def start(self, timeout: float = 5.0) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._run_loop, name="gaze-ws", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError("WebSocket server did not start in time")

    def stop(self) -> None:
        if self._loop and self._loop.is_running() and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
