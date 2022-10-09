"""Scarpe all leagues and its prize pools"""

import os
import sys; sys.path.append('../')

import time
import json
import asyncio

from utils.wrappers import OpendotaWrapper, SteamWrapper


async def prize_pool(leagues: set, batch_size: int = 64) -> dict:
    steam = SteamWrapper()
    league_prize_pool = {}
    tasks = [steam.fetch_tournament_prize_pool(leagueid) for leagueid in leagues]
    for batch in range(0, len(tasks), batch_size):
        print('batch :', batch)

        start_time = time.time()
        done = await asyncio.gather(*tasks[batch:batch+batch_size])
        for item in done:
            if isinstance(item, Exception):
                print(item)
            else:
                league_prize_pool[item['result']['league_id']] = item['result']['prize_pool']
                
        elapsed_time = time.time() - start_time
        if elapsed_time < 20:
            time.sleep(20-elapsed_time)

    return league_prize_pool


async def main():
    # ----------------------------------------------------------------------- #
    od = OpendotaWrapper()
    leagues = await od.fetch_leagues()
    leagues_id = [l['leagueid'] for l in leagues]
    with open('output/leagues.json', 'w', encoding='utf-8') as f:
        json.dump(leagues, f, ensure_ascii=False, indent=4)

    # ----------------------------------------------------------------------- #
    try:
        # Parse new leagues
        with open('output/prize_pools.json', 'r', encoding='utf-8') as f:
            league_prize_pool: dict = json.load(f)

        leagues_id = set([l for l in leagues_id if str(l) not in league_prize_pool])
        _league_prize_pool = await prize_pool(leagues_id)
        league_prize_pool.update(_league_prize_pool)

    except FileNotFoundError:
        # Parse from scratch
        league_prize_pool = await prize_pool(set(leagues_id))

    # ----------------------------------------------------------------------- #
    with open('output/prize_pools.json', 'w', encoding='utf-8') as f:
        json.dump(league_prize_pool, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

    