"""Scarpe all leagues and its prize pools"""

import os
import sys; sys.path.append('../')
import time
import json
import asyncio

from utils.wrappers import OpendotaWrapper
from utils.development import SessionHelper


os.environ['steam_api_key'] = '61886694FAAE6A81CD7E6337B6D98361'

async def prize_pool(leagues: set, batch_size: int = 120) -> dict:
    session = SessionHelper()
    api_key = os.environ.get('steam_api_key')

    league_prize_pool = {}
    tasks = [session.get(f"http://api.steampowered.com/IEconDOTA2_570/GetTournamentPrizePool/v1?key={api_key}&leagueid={leagueid}") for leagueid in leagues]
    for batch in range(0, len(tasks), batch_size):
        print('batch :', batch)

        start_time = time.time()
        done = await asyncio.gather(*tasks[batch:batch+batch_size])
        for item in done:
            json, code = item
            if code//100 == 2:
                league_prize_pool[json['result']['league_id']] = json['result']['prize_pool']
                
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            time.sleep(60-elapsed_time)

    await session.close()

    return league_prize_pool


async def main():
    od = OpendotaWrapper()
    leagues = await od.fetch_leagues()
    with open('output/leagues.json', 'w', encoding='utf-8') as f:
        json.dump(leagues, f, ensure_ascii=False, indent=4)

    league_prize_pool = await prize_pool(set([l['leagueid'] for l in leagues]))
    with open('output/prize_pools.json', 'w', encoding='utf-8') as f:
        json.dump(league_prize_pool, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

    