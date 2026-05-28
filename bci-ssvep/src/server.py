import asyncio
from ws.websockets import start, broadcast
from ws.http import start_http


async def main():
    await asyncio.gather(
        start(),
        start_http(broadcast)
    )

if __name__ == "__main__":
    asyncio.run(main())
