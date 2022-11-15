"""Parse MongoDB matches and convert them into property standart"""

import sys
import time
import pathlib
import traceback
if __name__ == '__main__':
    SCRIPT_DIR = pathlib.Path(__file__).parent
    sys.path.append(str(SCRIPT_DIR.parent))

import pymongo

import utils


client: pymongo.MongoClient

def reopen():
    global client

    try: client.close()
    except: pass
    client = pymongo.MongoClient("mongodb://localhost:27017/")


def parse_table(table, save_table, months, verbose=True):
    reopen()

    client['ParsedMatches'][save_table].delete_many({})
    _matches = client['DotaMatches'][table].find(
        {"start_time": {"$gt": int(time.time()) - 60*60*24*30*months}}
    )
    for idx, match in enumerate(_matches):    
        try:
            _match = od(match)
            _match = _match.dict()
            client['ParsedMatches'][save_table].insert_one(_match)
        except Exception as e:
            if str(e) == "'null'": traceback.print_exc()
            print(match['match_id'], e)

        if verbose and (idx+1) % 500 == 0:
            print(idx)
    


if __name__ == "__main__":
    reopen()

    od = utils.parsers.OpendotaParser()
    parse_table(
        table='leagueMatches', 
        save_table='leagueMatches', 
        months=48, verbose=True,
    )
    parse_table(
        table='publicMatches', 
        save_table='publicMatches', 
        months=48, verbose=True,
    )  
    # How to simple convert to pandas
    # matches = [_match for i in range(100_000)]
    # df = pd.DataFrame([m.__dict__ for m in matches])
    # df