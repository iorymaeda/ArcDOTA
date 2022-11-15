import sys
import logging
import pathlib
import asyncio
if __name__ == '__main__':
    SCRIPT_DIR = pathlib.Path(__file__).parent
    sys.path.append(str(SCRIPT_DIR.parent))

import pymongo

from utils import wrappers
from utils import _typing, exceptions



class Config:
    batch_size = 16
    # Time between batches
    timeout = 4
    start_id = 30_000
    end_id = 61_456
    write_in_file = False

def __log(*args):
    s = ""
    for a in args:
        s += str(a)
        s += " "
    if Config.write_in_file: logging.info(s)
    print(s)

def __mean(l: list[float|int]):
    return sum(l)/len(l)

def __exctract_time(odds: dict[str, _typing.hawk._odd]) -> int:
    if odds:
        created_at = __mean([odds[book_name][0]['created_at'] for book_name in odds])
        end_at = __mean([odds[book_name][-1]['created_at'] for book_name in odds])
        return created_at, end_at
    else:
        raise Exception("No odds there")

def __exists(id: int) -> bool:
    if table.find_one({'id': id}):
        return True
    else:
        return False

async def __parse_match(id: int) -> _typing.hawk.Match:
    try:
        match = await hawk.parse_match(id)
        created_at, end_at = __exctract_time(match["odds"])
        match['created_at'] = created_at
        match['end_at'] = end_at
        __log(f"match_{id=} parsed")
        return match
    except Exception as e:
        __log(f"match_{id=} got error: {e}")
        raise exceptions.hawk.CantParseMatch

async def parse_hawk():
    tasks = [__parse_match(match_id) for match_id in range(Config.start_id, Config.end_id+1)]
    for batch in range(0, len(tasks), Config.batch_size):
        print(f"Execute {batch} batch from: {len(tasks)}")

        done = await asyncio.gather(*tasks[batch:batch+Config.batch_size], return_exceptions=True)
        games = [item for item in done if not isinstance(item, Exception)]
        
        if games:
            try: table.insert_many(games)
            except Exception as e:
                __log(f"Error while insert many games: {e}")

        await asyncio.sleep(Config.timeout)

if __name__ == "__main__":
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client['Odds']
    table = db['hawkRaw']

    hawk = wrappers.HawkWrapper()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(parse_hawk())
    loop.close()
