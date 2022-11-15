"""Script to parse league games"""

import sys
import time
import asyncio
import pathlib
import traceback
if __name__ == '__main__':
    SCRIPT_DIR = pathlib.Path(__file__).parent
    sys.path.append(str(SCRIPT_DIR.parent))

import pymongo

import utils




def exists(match_id: int, table: pymongo.MongoClient) -> bool:
    if table.find_one({'match_id': match_id}): return True
    else: return False


async def main():
    raise NotImplementedError("There are bugs, fix it")
    months = 24
    batch_size = 60

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client['DotaMatches']
    od = utils.wrappers.OpendotaWrapper()

    # Scarpe all games
    matches = await od.fetch_league_games(start_time=time.time() - 60*60*24*30*months)
    print(f"Num of scarped games: {len(matches)}")

    # Insert in bd
    table = db['league']
    ids = set([match["match_id"] for match in table.find()])
    _matches = [match for match in matches if match['match_id'] not in ids]
    if _matches: table.insert_many(_matches)

    # Puge existing matches
    tasks = [od.fetch_game(match_id=match['match_id']) for match in matches if not exists(match_id=match['match_id'], table=db['leagueMatches'])]
    print(f"Num of cleaned games: {len(tasks)}")

    table = db['leagueMatches']
    for batch in range(0, len(tasks), batch_size):
        print(f"Execute {batch} batch. from : {len(tasks)}")

        done = await asyncio.gather(*tasks[batch:batch+batch_size], return_exceptions=True)
        games = [item for item in done if not isinstance(item, Exception)]
        [print(item) for item in done if isinstance(item, Exception)]

        if games:
            try: table.insert_many(games)
            except Exception:
                traceback.print_exc()

        # for item in done:
        #     if not isinstance(item, Exception):
        #         try: db['leagueMatches'].find_one_and_replace(filter = {'match_id': item['match_id']}, replacement=item, upsert=True)
        #         except Exception:
        #             traceback.print_exc()



async def main_slow():
    months = 48
    batch_size = 60

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client['DotaMatches']
    od = utils.wrappers.OpendotaWrapper()

    # Scarpe all games
    matches = await od.fetch_league_games(start_time=time.time() - 60*60*24*30*months)
    print(f"Num of scarped games: {len(matches)}")

    # Insert in bd
    table = db['league']
    ids = set([match["match_id"] for match in table.find()])
    _matches = [match for match in matches if match['match_id'] not in ids]
    if _matches: table.insert_many(_matches)

    # Puge existing matches
    ids = [match['match_id'] for match in matches if not exists(match_id=match['match_id'], table=db['leagueMatches'])]
    print(f"Num of cleaned games: {len(ids)}")

    table = db['leagueMatches']
    for match_id in ids:
        try:
            game = await od.fetch_game(match_id=match_id)
            table.insert_one(game)
        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_slow())
    loop.close()

    