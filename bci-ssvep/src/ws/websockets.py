import asyncio
from websockets.asyncio.server import serve

connected_clients: set = set()


async def handler(websocket):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            print(f"Received: {message}")
    finally:
        connected_clients.discard(websocket)


async def broadcast(message: str):
    if connected_clients:
        await asyncio.gather(*[ws.send(message) for ws in connected_clients])


async def start():
    async with serve(handler, "", 8765) as server:
        await server.serve_forever()
