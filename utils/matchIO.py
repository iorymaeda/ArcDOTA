"""

    This is trash legacy migrated from old sources!!!
    
    
"""

raise NotImplementedError

import aiohttp
import asyncio
import datetime
import time 
import pymongo
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd


class MatchIO():
    def __init__(self, match_id: int):
        self.match_id = match_id
        self.live = None
        self.match = None
        self.heroes_picked = None

        self.D = loadModelScaler.load(load_model=True, load_scaler=True)
        self.scaler = self.D['scaler']
        self.model_builded = False

        timeout = aiohttp.ClientTimeout(total=65)
        self.session = aiohttp.ClientSession(timeout=timeout)

        self.client = pymongo.MongoClient("mongodb://localhost:27017/")    
        self.table = self.client['DotaMatches']['leagueMatches']

        self.RADIANT_SLOTS = [0, 1, 2, 3, 4]
        self.DIRE_SLOTS = [128, 129, 130, 131, 132]

        self.radiant_players = None
        self.radiant_heroes  = None
        self.radiant_team    = None
        
        self.dire_players = None
        self.dire_heroes  = None
        self.dire_team    = None
        
        self.league = None
        
        self.X: list = None


    async def predict(self):   
        """Returns int as prob team1 to win (from 0 to 1) or str due errors""" 
        print('predict!!!')
        if self.X is None:
            await self.collectData()

        if not self.model_builded:
            # First build a graph
            self.D['model'](self.X)
            # Then load weights
            self.D['model'].load_weights(self.D['weights'])
            self.model_builded = True

        pred = self.D['model'].predict(self.X)
        return pred[0, 0]


    async def collectData(self):
        time_test = time.time()
        await self.getMatchInfo()
        await self.getData()
        print('Get games took {} sec'.format(time.time() - time_test))


    def currentSec(self) -> int:
        date = datetime.datetime.now()
        return date.second


    async def get_live(self) -> list[dict, bool]:
        json, status = await self.sessionGetPost(
            URL=f'http://api.steampowered.com/IDOTA2Match_570/GetLiveLeagueGames/v1?key={config.STEAM_API}&match_id={self.match_id}',
            od_api=False
        )
        if json['result']['games']:
            self.live = True
            self.match = json['result']['games'][0]
        else:
            self.live = False
     
    
    async def apiLimit(self, seconds: int, headers: str):
        if int(headers['x-rate-limit-remaining-month']) > 0:
            if int(headers['x-rate-limit-remaining-minute']) <= 0:
                print('minute limit is over') 
                time.sleep(60-seconds + 1)
                return True
        else:
            print('month limit is over')    
        
        
    async def sessionGetPost(self, URL: str, get = True, counter = 0, od_api = True) -> list[dict, int]:
        json = None
        status = 666
        if counter > 3: return json, status
        try:
            current_sec = self.currentSec()
            response = await self.session.get(URL) if get else await self.session.post(URL)
            status = response.status
            if status//100 == 2:
                json = await response.json()
                return json, status
            else:
                if od_api:
                    is_api = await self.apiLimit(current_sec, response.headers)
                # If this api limits error, don't increase counter
                # Else this error or something - increase.
                json, status = await self.sessionGetPost(URL=URL, get=get, counter=counter if is_api else counter+1, od_api=od_api)
                return json, status
        # Except timeout error
        except Exception as e:
            json, status = await self.sessionGetPost(URL=URL, get=get, counter=counter+1, od_api=od_api)
            return json, status


    def load_match_from_bd(self, match_id: int) -> dict:
        """ returns None if not founded"""
        return self.table.find_one({'match_id': match_id})

    
    async def parse(self, match_id: int):
        URL = f'https://api.opendota.com/api/request/{match_id}'
        await self.sessionGetPost(URL, get = False)

        
    def save_match(self, match: dict):
        self.table.find_one_and_replace(filter = {'match_id': match['match_id']}, replacement = match, upsert = True)    

        
    async def get_game_from_api(self, match_id: int) -> list[dict, int]:
        async def get(URL, counter = 0):
            game, status = await self.sessionGetPost(URL)
            # WE NEED ONLY PARSED GAMES
            if counter > 30:
                return None, 666
            elif game:
                if game['objectives']:
                    return game, status
                
                # If game not parsed and past a lot of time
                # reutrn None, cause it will not parse
                elif time.time() - game['start_time'] > 86400: #86400 is one day
                    return None, 666
                
                print(game['match_id'], '\nWait :', counter, flush=True)
                
                await self.parse(match_id)
                await asyncio.sleep(16)
                
                return await get(URL, counter+1)
            else:
                # Can't load game
                return None, 666
            
        return await get(f'https://api.opendota.com/api/matches/{match_id}')


    async def get_match(self, match_id: int):
        match = self.load_match_from_bd(match_id)
        if match is None:
            match, status = await self.get_game_from_api(match_id)
            if match: self.save_match(match)
        return match
    
    
    async def get_team_games_from_api(self, team: int) -> list[dict, int]:
        # this API returns ALL team's games
        URL = f'https://api.opendota.com/api/teams/{team}/matches/'
        text, status = await self.sessionGetPost(URL)
        return text, status

    
    async def get_teams_games_from_api(self, team1: int, team2: int) -> list[dict, int]:
        URL = f"https://api.opendota.com/api/explorer?sql=SELECT%20%0Amatches.match_id%2C%0Amatches.start_time%2C%0Amatches.radiant_team_id%2C%0Amatches.dire_team_id%0AFROM%20matches%0AWHERE%20((RADIANT_TEAM_ID%20%3D%20{team1}%20and%20DIRE_TEAM_ID%20%3D%20{team2})%20or%20(RADIANT_TEAM_ID%20%3D%20{team2}%20and%20DIRE_TEAM_ID%20%3D%20{team1}))%0AORDER%20BY%20matches.start_time%20DESC"
        json, status = await self.sessionGetPost(URL)
        return json, status
    
    
    def buildng_counter(self, objectives) -> tuple[int, int]:
        r_twd_t = 0
        d_twd_t = 0
        for obj in objectives:
            if obj['type'] == 'building_kill':
                if obj['key'][9] == 'b': # radiant destroy tower
                    r_twd_t+=1
                elif obj['key'][9] == 'g': # dire destroy tower
                    d_twd_t+=1    
        return r_twd_t, d_twd_t

    
    def check_for_correctness(self, match: dict, t1, p1, t2, p2) -> bool:
        if match['objectives']:
            r_twd, d_twd = self.buildng_counter(match['objectives'])
            destroyed_towers = r_twd + d_twd
            total_score = match['radiant_score'] + match['dire_score']

            if match['duration'] > self.scaler.PREP_VALUES['time_low_drop'] and destroyed_towers >= self.scaler.PREP_VALUES['destroyed_towers_low_drop'] and total_score >= self.scaler.PREP_VALUES['total_score_low_drop']:
                radiant_players = [player['account_id'] for player in match['players'] if player['player_slot'] in self.RADIANT_SLOTS]
                radiant_team = match['radiant_team_id']

                dire_players = [player['account_id'] for player in match['players'] if player['player_slot'] in self.DIRE_SLOTS]
                dire_team = match['dire_team_id']

                radiant_players = set(radiant_players)
                dire_players = set(dire_players)

                if t2 is None:
                    if t1 == radiant_team:
                        if len(set(radiant_players) & set(p1)) < 4:
                            return False
                        return True

                    elif t1 == dire_team:
                        if len(set(dire_players) & set(p1)) < 4:
                            return False 
                        return True
                else:
                    if t1 == radiant_team and t2 == dire_team:
                        if len(set(radiant_players) & set(p1)) < 4 and len(set(dire_players) & set(p2)) < 4:
                            return False
                        return True
                    elif t2 == radiant_team and t1 == dire_team:
                        if len(set(radiant_players) & set(p2)) < 4 and len(set(dire_players) & set(p1)) < 4:
                            return False
                        return True
                    
                    
    async def collect_matches(self, team_matches, t1, p1, t2=None, p2=None) -> list:
        """Collect WINDOW team matches """
        GAMES = []
        uncorrected_matches = 0
        for match_id in team_matches:
            if len(GAMES) == self.scaler.PREP_VALUES['WINDOW'] or uncorrected_matches > 15:
                return GAMES
            match = await self.get_match(match_id)
            if match:
                if self.check_for_correctness(match, t1, p1, t2, p2):
                    GAMES.append(match)
                else:
                    uncorrected_matches+=1
        return GAMES


    async def getMatchInfo(self):
        await self.get_live()
        if self.live:
            self.radiant_players = [player['account_id'] for player in self.match['players'] if player['team'] == 0]
            self.radiant_heroes = [player['hero_id'] for player in self.match['players'] if player['team'] == 0]
            self.radiant_team = self.match['radiant_team']['team_id']
            self.dire_players = [player['account_id'] for player in self.match['players'] if player['team'] == 1]
            self.dire_heroes = [player['hero_id'] for player in self.match['players'] if player['team'] == 1]
            self.dire_team = self.match['dire_team']['team_id']
            self.league = self.match['league_id']
            if 0 in self.dire_heroes + self.radiant_heroes:
                self.heroes_picked = False
            else:
                self.heroes_picked = True
        else:
            self.match = await self.get_match(self.match_id)

            self.radiant_players = [player['account_id'] for player in self.match['players'] if player['player_slot'] in self.RADIANT_SLOTS]
            self.radiant_heroes = [player['hero_id'] for player in self.match['players'] if player['player_slot'] in self.RADIANT_SLOTS]
            self.radiant_team = self.match['radiant_team_id']
            self.dire_players = [player['account_id'] for player in self.match['players'] if player['player_slot'] in self.DIRE_SLOTS]
            self.dire_heroes = [player['hero_id'] for player in self.match['players'] if player['player_slot'] in self.DIRE_SLOTS]
            self.dire_team = self.match['dire_team_id']
            self.league =  self.match['leagueid']
            if 0 in self.dire_heroes + self.radiant_heroes:
                self.heroes_picked = False
            else:
                self.heroes_picked = True


    async def check_heroes(self):
        await self.get_live()
        if self.heroes_picked:
            self.X[ 8] = np.array([self.radiant_heroes], dtype='int32')
            self.X[16] = np.array([self.dire_heroes], dtype='int32')
            print('Live: {}, Heroes has been picked: {}'.format(self.live, self.heroes_picked))
            return True
            

    async def getGames(self):
        assert self.match != None

        radiant_team_matches, status = await self.get_team_games_from_api(self.radiant_team)
        assert status//100 == 2
        radiant_team_matches = [m['match_id'] for m in radiant_team_matches if m['match_id'] < self.match_id]
        radiant_team_matches.sort(reverse=True)

        dire_team_matches, status = await self.get_team_games_from_api(self.dire_team)
        assert status//100 == 2
        dire_team_matches = [m['match_id'] for m in dire_team_matches if m['match_id'] < self.match_id]
        dire_team_matches.sort(reverse=True)

        meetings_matches, status = await self.get_teams_games_from_api(self.radiant_team, self.dire_team)
        assert status//100 == 2
        meetings_matches = meetings_matches['rows']
        meetings_matches = [m['match_id'] for m in meetings_matches if m['match_id'] < self.match_id]
        meetings_matches.sort(reverse=True)

        GAMES_T1 = await self.collect_matches(
            radiant_team_matches, 
            self.radiant_team, 
            self.radiant_players
        )
        GAMES_T2 = await self.collect_matches(
            dire_team_matches, 
            self.dire_team, 
            self.dire_players
        )
        GAMES_M  = await self.collect_matches(
            meetings_matches, 
            self.radiant_team, 
            self.radiant_players, 
            self.dire_team, 
            self.dire_players
        )
        print(len(GAMES_T1), len(GAMES_T2), len(GAMES_M))
        return GAMES_T1, GAMES_T2, GAMES_M
    
    
    async def parse_prize_pool(self, leagues: set, batch_size: int = 8) -> dict:
        league_prize_pool = {}
        tasks = [self.sessionGetPost(f"http://api.steampowered.com/IEconDOTA2_570/GetTournamentPrizePool/v1?key={config.STEAM_API}&leagueid={leagueid}", od_api=False) for leagueid in leagues]
        for batch in range(0, len(tasks), batch_size):
            done = await asyncio.gather(*tasks[batch:batch+batch_size])
            for item in done:
                json, code = item
                if code//100 == 2:
                    league_prize_pool[json['result']['league_id']] = json['result']['prize_pool']
        return league_prize_pool


    async def end_session(self):
        self.client.close()
        await self.session.close()