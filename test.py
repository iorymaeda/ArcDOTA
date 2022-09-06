import asyncio

from utils.wrappers import SteamWrapper

async def main():
    steam = SteamWrapper()
    result = await steam.fetch_live_games(partner=0)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
 