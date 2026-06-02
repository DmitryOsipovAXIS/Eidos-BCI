import asyncio
from ws.websockets import start, broadcast
from ws.http import start_http
from realtime.pipeline import start_live, LiveCollectArgs


async def main():
    loop = asyncio.get_running_loop()
    args = LiveCollectArgs(no_eeg=True)

    await asyncio.gather(
        start(),
        start_http(broadcast),
        loop.run_in_executor(None, lambda: start_live(args, broadcast, loop)),
    )

if __name__ == "__main__":
    asyncio.run(main())
