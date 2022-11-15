import sys
import time
import pathlib
if __name__ == '__main__':
    SCRIPT_DIR = pathlib.Path(__file__).parent
    sys.path.append(str(SCRIPT_DIR.parent))

import pymongo
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

import utils
from utils._typing import hawk
from utils.development import suppress

RADIANT_SIDE = utils.base.DotaconstantsBase.RADIANT_SIDE
DIRE_SIDE = utils.base.DotaconstantsBase.DIRE_SIDE

def find_by_drafts(r_h_heroes, d_h_heroes, match_time) -> list[int]:
    output = []
    
    b = (start_time < (match_time + 3600*12)) & (start_time > (match_time - 3600*24))
    for r, d, m_id, win in zip(radiant_heroes[b], dire_heroes[b], match_id[b], radiant_win[b]):
        if r_h_heroes == r and d_h_heroes == d: 
            output.append( (m_id, win) )
            
    return output

def mean(l: list[float|int]):
    return sum(l)/len(l)

def exctract_time(odds: dict[str, hawk._odd]) -> int:
    if odds:
        created_at = mean([odds[book_name][0]['created_at'] for book_name in odds])
        end_at = mean([odds[book_name][-1]['created_at'] for book_name in odds])
        return int(mean([created_at, end_at])) + 21600
    else:
        raise Exception("No odds there")

def get_prematch_matches():
    table = client['Odds']['hawkParsed']
    _hawk_matches = list(table.find())

    table = client['Odds']['hawkRaw']
    _hawk_matches += list(table.find())
    
    print(f'num of games from db:', len(_hawk_matches))
    # Remove duplicates
    hawk_ids = []
    hawk_matches: list[hawk.Match] = []
    for match in _hawk_matches:
        if match['id'] not in hawk_ids:
            hawk_matches.append(match)
            hawk_ids.append(match['id'])

    print(f'num of games after remove duplicates:', len(hawk_matches))
    return hawk_matches

def parse_raw():
    parsed_matches = []
    hawk_matches = get_prematch_matches()
    for hawk_match in hawk_matches:
        with suppress(Exception, print_exc=False):
            r_h_heroes = set(hawk_match['radiant_heroes'])
            d_h_heroes = set(hawk_match['dire_heroes'])

            odds_time = exctract_time(hawk_match['odds'])

            founded_match = find_by_drafts(r_h_heroes, d_h_heroes, odds_time)
            
            if len(founded_match) > 1:
                raise "Too many games"
                
            if len(founded_match) == 0:
                raise "Games not found"
            
            founded_match = founded_match[0]
            for book_name in hawk_match['odds']:
                if not hawk_match['odds'][book_name][0]['live']:
                    match = {}
                    match['match_id'] = founded_match[0]
                    match['radiant_win'] = founded_match[1]
                    match['book_name'] = book_name

                    match['r_odd'] = hawk_match['odds'][book_name][0]['r_odd']
                    match['d_odd'] = hawk_match['odds'][book_name][0]['d_odd']

                    parsed_matches.append(match)

    odds_df = pd.DataFrame(parsed_matches)
    odds_df.to_csv(SCRIPT_DIR / 'output/prematch_odds.csv', index=False)

if __name__ == "__main__":
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    parser = utils.parsers.PropertyParser()

    table = client['ParsedMatches']['leagueMatches']
    df = parser(list(table.find()))

    match_id = df['match_id'].values
    start_time = df['start_time'].values
    radiant_win = df['radiant_win'].values

    radiant_heroes = df[[f'{_}_hero_id' for _ in RADIANT_SIDE]].values
    dire_heroes = df[[f'{_}_hero_id' for _ in DIRE_SIDE]].values

    radiant_heroes = np.array([set(d) for d in radiant_heroes])
    dire_heroes = np.array([set(d) for d in dire_heroes])

    parse_raw()