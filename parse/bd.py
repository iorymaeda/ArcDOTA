"""Parse MongoDB matches and convert them into property standart"""

# importing modules from parent folder
import sys; sys.path.append('../')
import time

import pymongo

import utils


client: pymongo.MongoClient

def reopen():
    global client

    try: client.close()
    except: pass
    client = pymongo.MongoClient("mongodb://localhost:27017/", maxPoolSize=50)


def parse_table(table, save_table, months, verbose=True):
    reopen()
    _matches = client['DotaMatches'][table].find(
        {
            "start_time": {
                "$gt": int(time.time()) - 60*60*24*30*months
            }
        }
    )
    
    for idx, match in enumerate(_matches):    
        try:
            _match = od(match)
            _match = _match.dict()
            client['ParsedMatches'][save_table].find_one_and_replace(
                filter={'match_id': _match['match_id']}, 
                replacement=_match, 
                upsert=True,
            )
        except Exception as e:
            errors.append(e)
            error_matches.append(match['match_id'])
            
        if (idx+1) % 1000 == 0:
            # reopen()
            pass

        if verbose and (idx+1) % 500 == 0:
            print(idx)


if __name__ == "__main__":
    reopen()
    od = utils.parsers.OpendotaParser(
        dotaconstants_path='../scarpe/dotaconstants',
        leagues_path='../scarpe/output/leagues.json', 
        prize_pools_path='../scarpe/output/prize_pools.json',
    )

                
    errors = []
    error_matches = []
    parse_table(
        table='leagueMatches', 
        save_table='leagueMatches', 
        months=30, verbose=False,
    )
    parse_table(
        table='publicMatches', 
        save_table='publicMatches', 
        months=30, verbose=False,
    )  
    # How to simple convert to pandas
    # matches = [_match for i in range(100_000)]
    # df = pd.DataFrame([m.__dict__ for m in matches])
    # df