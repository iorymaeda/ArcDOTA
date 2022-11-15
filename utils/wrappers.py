import os
import time
import json
import asyncio
import datetime
from typing import Literal

import pymongo
import aiohttp
import aiofiles
import pandas as pd
from lxml import html

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from . import _typing, parsers, evaluator, tokenizer, scalers, exceptions
from .time_series import PrematchTSCollector
from .base import ConfigBase, DotaconstantsBase
from .development import OpendotaSession, SessionHelper, suppress, substitute, execute_cor


class SeleniumWrapper(uc.Chrome):
    def __init__(self):
        super().__init__(use_subprocess=False)
        self.loop = asyncio.get_event_loop()

    async def async_get(self, url):
        f = self.loop.run_in_executor(None, self.get, url)
        while not f.done(): await asyncio.sleep(0.1)
        return self.page_source

    def wait(self, elem: str, wait_time: int = 4):
        try:
            WebDriverWait(self, 1).until(EC.presence_of_element_located((By.ID, elem)))
            return True
        except TimeoutException:
            if wait_time <= 0:
                return False
            else:
                return self.wait(elem, wait_time-1)


class HawkWrapper(SessionHelper, DotaconstantsBase):
    BASE_URL = "https://hawk.live"

    def __init__(self, use_selenium=True):
        super().__init__()
        self.npc_to_id, self.id_to_hero = self._load_heroes()

        self.use_selenium = use_selenium
        if self.use_selenium:
            self.selenium = SeleniumWrapper()
            self.selenium.get(self.BASE_URL)
            if not self.selenium.wait('app', 30):
                raise exceptions.hawk.MainPageNotFound('Cant get hawk main page')

    async def fetch_match(self, match_id: int) -> aiohttp.client.ClientResponse | str:
        url = f'{self.BASE_URL}/matches/{match_id}'
        if self.use_selenium:
            self.selenium.get(url)
            if not self.selenium.wait('app', 3):
                raise exceptions.hawk.MatchNotFound('Cant get hawk match')
            return self.selenium.page_source
        
        response = await self.get(url, raw=True)
        if response.status//100 not in [2, 3]:
            raise exceptions.hawk.MatchNotFound('Cant get hawk match')
        return response

    async def fetch_main_page(self) -> aiohttp.client.ClientResponse | str:
        url = f'{self.BASE_URL}/'
        if self.use_selenium:
            self.selenium.get(url)
            if not self.selenium.wait('app', 3):
                raise exceptions.hawk.MainPageNotFound('Cant get hawk main page')
            return self.selenium.page_source

        response = await self.get(url, raw=True)
        if response.status//100 not in [2, 3]:
            raise exceptions.hawk.MainPageNotFound('Cant get hawk main page')
        return response

    @substitute(Exception, surrogate=exceptions.hawk.CantParseMatch)
    async def parse_match(self, match_id: int) -> _typing.hawk.Match:
        response = await self.fetch_match(match_id=match_id)
        response = response if isinstance(response, str) else await response.text() 

        tree: html.HtmlElement = html.fromstring(response)
        data_elems: list[html.HtmlElement] = tree.xpath('//*[@id="app"]')

        if len(data_elems) > 1: raise Exception('Unexcepted hawk html tree while fetch match')

        data = data_elems[0]
        data = dict(data.attrib)
        data = json.loads(data['data-page'])

        is_team1_radiant = data['props']['match']['is_team1_radiant']

        radiant_heroes = [ self.npc_to_id[hero['hero']['code_name']] for hero in data['props']['match']['picks'] if hero['is_radiant'] ]
        dire_heroes = [ self.npc_to_id[hero['hero']['code_name']] for hero in data['props']['match']['picks'] if not hero['is_radiant'] ]

        match_data = data['props']['match']
        radiant_team = match_data['team1']
        dire_team = match_data['team2']
        is_radiant_won = match_data['is_radiant_won']
        championship_name = match_data['championship_name'] if 'championship_name' in match_data else ''

        match_odds_info_array = {}
        for odds in data['props']['match_odds_info_array']:
            _odds = []
            for odd in odds['odds']:
                with suppress(ValueError):
                    if is_team1_radiant == True:
                        if odds['is_team1_first'] == True:
                            radiant_odd = float(odd['first_team_winner'])
                            dire_odd = float(odd['second_team_winner'])
                            
                        elif odds['is_team1_first'] == False:
                            radiant_odd = float(odd['second_team_winner'])
                            dire_odd = float(odd['first_team_winner'])
                        else: raise
                        
                    elif is_team1_radiant == False:
                        if odds['is_team1_first'] == True:
                            dire_odd = float(odd['first_team_winner'])
                            radiant_odd = float(odd['second_team_winner'])
                            
                        elif odds['is_team1_first'] == False:
                            dire_odd = float(odd['second_team_winner'])
                            radiant_odd = float(odd['first_team_winner'])         
                        else: raise
                        
                    else: raise
                    
                    created_at = int(time.mktime(datetime.datetime.strptime(odd['created_at'], '%Y-%m-%d %H:%M:%S').timetuple()))
                    _odds.append({
                        'r_odd': radiant_odd,
                        'd_odd': dire_odd,
                        'live': False,
                        'created_at': created_at
                    })
                    
            match_odds_info_array[odds['odds_provider_code_name']] = _odds

        if not is_team1_radiant: 
            radiant_heroes, dire_heroes = dire_heroes, radiant_heroes
            radiant_team, dire_team = dire_team, radiant_team

        return {
            "id": match_id,
            "match_id": [],
            "championship_name": championship_name,
            "is_radiant_won": is_radiant_won,
            "radiant_team": radiant_team,
            "dire_team": dire_team,
            "radiant_heroes": radiant_heroes,
            "dire_heroes": dire_heroes,
            "odds": match_odds_info_array
        }

    @substitute(Exception, surrogate=exceptions.hawk.CantParseSeries)
    async def parse_series(self) -> list[_typing.hawk.Series]:
        response = await self.fetch_main_page()
        response = response if isinstance(response, str) else await response.text() 

        tree: html.HtmlElement = html.fromstring(response)
        data_elems: list[html.HtmlElement] = tree.xpath('//*[@id="app"]')

        if len(data_elems) > 1: raise Exception('Unexcepted hawk html tree while fetch match')

        data = data_elems[0]
        data = dict(data.attrib)
        data = json.loads(data['data-page'])
        return data['props']['series']


class SteamWrapper(SessionHelper, ConfigBase):
    BASE_URL = "http://api.steampowered.com"

    def __init__(self):
        super().__init__()    
        self.key = os.environ.get('steam_api_key')
        self.key = '' if self.key is None else f"key={self.key}"


    async def fetch_live_league_game(self, match_id: int) -> _typing.steam.LeagueGame:
        url = f"{self.BASE_URL}/IDOTA2Match_570/GetLiveLeagueGames/v1?{match_id=}"
        url = url + f"&{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise exceptions.steam.SteamError(status, data)

        data: _typing.steam.GetLiveLeagueGames
        if data['result']['games']:
            return data['result']['games'][0]
        raise exceptions.steam.LiveGameNotFound(f"Game not found")
        

    async def fetch_live_league_games(self, partner:Literal[0, 1, 2, 3]=0) -> list[_typing.steam.LeagueGame]:
        url = f"{self.BASE_URL}/IDOTA2Match_570/GetLiveLeagueGames/v1?{partner=}"
        url = url + f"&{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise exceptions.steam.SteamError(status, data)

        data: _typing.steam.GetLiveLeagueGames
        if data['result']['games']:
            return data['result']['games']
        raise exceptions.steam.LeagueGamesNotFound


    async def fetch_live_games(self, partner:Literal[0, 1, 2, 3]=0) -> _typing.steam.GetLiveLeagueGames:
        url = f"{self.BASE_URL}/IDOTA2Match_570/GetLiveLeagueGames/v1?{partner=}"
        url = url + f"&{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise exceptions.steam.SteamError(status, data)
        return data


    async def fetch_tournament_prize_pool(self, leagueid:int) -> _typing.steam.GetTournamentPrizePool:
        url = f"http://api.steampowered.com/IEconDOTA2_570/GetTournamentPrizePool/v1?{leagueid=}"
        url = url + f"&{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise exceptions.steam.PrizePoolNotFound(status, leagueid)

        if data['result']['league_id'] == leagueid: return data
        raise exceptions.steam.PrizePoolNotFound(status, leagueid)


    async def download_replay(self, url: str, save_path: str) -> bool:
        async with aiohttp.ClientSession(timeout = aiohttp.ClientTimeout(total = 0, connect = 25)) as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    f = await aiofiles.open(save_path, mode='wb')
                    await f.write(await resp.read())
                    await f.close()
                    return True


class OpendotaWrapper(OpendotaSession):
    BASE_URL = "https://api.opendota.com/api"

    def __init__(self):
        super().__init__()
        self.key = os.environ.get('opendota_api_key')
        self.key = '' if self.key is None else f"api_key={self.key}"


    async def parse_game(self, match_id: int):
        """League games usually parses automaticly.
        Game parsing may take up to 15 minutes"""
        url = f'{self.BASE_URL}/request/{match_id}' 
        url = url + f"?{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise exceptions.opendota.OpendotaError(status, data)


    async def validate_match(self, match: _typing.opendota.Match):
        """Validate match as parsed or not"""
        return True if match['objectives'] else False


    async def fetch_game(self, match_id: _typing.opendota.match_id) -> _typing.opendota.Match:
        url = f'{self.BASE_URL}/matches/{match_id}'
        url = url + f"?{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise exceptions.opendota.OpendotaError(status, data)

        if not await self.validate_match(match=data):
            # If the game is not parsed and has passed a lot of time it will not be parsed
            # 86400 is one day
            if data['leagueid'] == 0:
                # Public match, here are no auto parse
                if data['start_time'] > 86400 * 8:
                    raise exceptions.opendota.ParsingNotPossible
            else:
                # League match, here are auto parse,
                # So if game after one day still not parsed - broken
                if data['start_time'] > 86400:
                    raise exceptions.opendota.ParsingNotPossible

            raise exceptions.opendota.GameNotParsed
        return data


    async def force_fetch_game(self, match_id: _typing.opendota.match_id, num=20) -> _typing.opendota.Match | None:
        """Parse and fetch game"""
        num-= 1
        if num <= 0: 
            raise exceptions.opendota.ParsingNotPossible

        try:
            game = await self.fetch_game(match_id=match_id)
            return game
        except exceptions.opendota.GameNotParsed:
            await self.parse_game(match_id=match_id)
            await asyncio.sleep(30)
            print('GameNotParsed: Try another one force fetch...')
            return await self.force_fetch_game(match_id, num=num) 
        
        except exceptions.opendota.OpendotaError:
            await asyncio.sleep(4)
            print('OpendotaError: Try another one force fetch...')
            return await self.force_fetch_game(match_id, num=num)
            

    async def fetch_teams(self) -> list[_typing.opendota.Team]:
        url = f'{self.BASE_URL}/teams'
        url = url + f"?{self.key}" if self.key else url
        resp, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise exceptions.opendota.OpendotaError(status, resp)
        return resp


    async def fetch_team_games(self, team: int) -> _typing.opendota.TeamMatches:
        url = f'{self.BASE_URL}/teams/{team}/matches'
        url = url + f"?{self.key}" if self.key else url
        resp, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise exceptions.opendota.OpendotaError(status, resp)
        return resp

    
    async def fetch_teams_meetings(self, team1: int, team2: int, limit:int=None) -> list[int]:
        if limit is None: limit = 1_000
        url = f"""{self.BASE_URL}/explorer?sql=
        SELECT%20%0Amatches.match_id%2C%0Amatches.start_time%2C%0Amatches.radiant_team_id%2C%0Amatches.dire_team_id%0A
        FROM%20matches%0AWHERE%20((RADIANT_TEAM_ID%20%3D%20{team1}%20and%20DIRE_TEAM_ID%20%3D%20{team2})%20or%20(RADIANT_TEAM_ID%20%3D%20{team2}%20and%20DIRE_TEAM_ID%20%3D%20{team1}))%0A
        ORDER%20BY%20matches.start_time%20DESC%0ALIMIT%20{200}"""
        url = url + f"&{self.key}" if self.key else url
        games, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise exceptions.opendota.OpendotaError(status, games)

        games = games['rows']
        return [m['match_id'] for m in games]


    async def fetch_teams_games(self, team1: int, team2: int) -> _typing.opendota.TeamMatches:
        return await self.fetch_teams_meetings(team1=team1, team2=team2)


    async def fetch_public_games(self, start_time: int, avg_mmr: int, limit: int = None) -> _typing.opendota.SQLPublicMatches:
        if limit is None: limit = 100_000
        date = datetime.datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d")
        url = f"""{self.BASE_URL}/explorer?sql=
            SELECT%20*%0AFROM%20public_matches%0AWHERE%20TRUE%0AAND%20
            public_matches.start_time%20%3E%3D%20
            extract(epoch%20from%20timestamp%20%27{date}%27)%0AAND%20
            public_matches.AVG_MMR%20%3E%3D%20{avg_mmr}%0ALIMIT%20{limit}"""
        url = url + f"&{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise exceptions.opendota.OpendotaError(status, data)
        return data['rows']


    async def fetch_league_games(self, start_time: int, limit: int = None) -> _typing.opendota.SQLLeagueMatches:
        if limit is None: limit = 100_000
        date = datetime.datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d")
        url = f"""{self.BASE_URL}/explorer?sql=
            SELECT%20%0Amatches.match_id%2C%0Aleagues.name%20leaguename%2C%0Aleagues.tier%20leaguetier%2C%0Amatches.start_time%2C%0Amatches.lobby_type%2C%0Amatches.human_players%2C%0Amatches.game_mode%2C%0Amatches.radiant_team_id%2C%0Amatches.dire_team_id%2C%0Amatches.radiant_captain%2C%0Amatches.dire_captain%0A
            FROM%20matches%0AJOIN%20leagues%20using(leagueid)%0A
            WHERE%20matches.start_time%20%3E%3D%20extract(epoch%20from%20timestamp%20%27{date}%27)%0A
            ORDER%20BY%20matches.start_time%20DESC%0ALIMIT%20{limit}"""
        url = url + f"&{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise exceptions.opendota.OpendotaError(status, data)
        return data['rows']


    async def scarpe_games(self, match_ids: list[_typing.opendota.match_id], batch_size=60):
        raise NotImplementedError
        games = []
        tasks = [self.force_fetch_game(match_id) for match_id in match_ids]
        for batch in range(0, len(tasks), batch_size):
            done = await asyncio.gather(*tasks[batch:batch+batch_size])


    async def fetch_leagues(self) -> list[_typing.opendota.League]:
        url = f"{self.BASE_URL}/leagues"
        url = url + f"?{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]: raise exceptions.opendota.OpendotaError(status, data)

        return data


    async def fetch_team_stack(self, team: int) -> list[int]:
        url = f'{self.BASE_URL}/teams/{team}/players'
        url = url + f"?{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise exceptions.opendota.OpendotaError(status, data)
        
        data: list[_typing.opendota.TeamPlayers]
        return [player['account_id'] for player in data if player['is_current_team_member']]


class PrematchWrapper(ConfigBase):
    """Scarpe, parse and prepare data to feed to prematch model

    Be careful: ConfigBase after load model in inference will output config from train phase"""

    def __init__(self, 
        scaler_path:str = "inference/files/scaler_league.pkl",
        tokenizer_path:str = "inference/files/tokenizer_league.pkl"
        ):
        self.opendota_wrapper = OpendotaWrapper()
        self.steam_wrapper = SteamWrapper()

        path = self._get_relative_path()
        path = path.parent.resolve()
        self.opendota_parser = parsers.OpendotaParser()
        self.property_parser = parsers.PropertyParser()
        self.evaluator = evaluator.Evaluator()
        self.scaler = scalers.DotaScaler(path=path / scaler_path)

        self.tokenizer_path = path / tokenizer_path
        # //TODO: fix tokenizer, put it together with a model
        self.tokenizer = tokenizer.Tokenizer(path=self.tokenizer_path)
        # //TODO: put to configs mask_type, y_output and anothers hyper-parameters 
        self.batch_size = 60

        client = pymongo.MongoClient("mongodb://localhost:27017/") 
        self.table = client['DotaMatches']['leagueMatches']
        self.loop = asyncio.get_event_loop()
        
    async def log(self, *args):
        message = ''.join(map(str, args))
        print(message)

    async def collect_prematch(self, window_size: int, team1: int|None = None, team2: int|None = None, match_id: int|None = None, league_id: int|None = None, prize_pool: int|None = None) -> tuple[int, int, pd.DataFrame, pd.DataFrame]:
        """Collect games and returns `corpus`, `anchor`
        
        Args:
            - window_size: int - how many games of both teams should be in `corpus`. You must provide this manualy due ensemble modeling with different configuration, so put there highest window size
            - team1: int - the first team id, may be `None` if `match_id` provided
            - team2: int - the second team id, may be `None` if `match_id` provided
            - match_id: int - the match's id you need to predict, if provided `anchor` will collect from this match, also match that played after this one will not includes in `corpus`
            - league_id: int - the match's league id, may be `None` if `prize_pool` provided
            - prize_pool: int - the match's prize pool, may be `None` if `league_id` provided

        Returns:
            - team1: int - first team id
            - team2: int - second team id
            - corpus: pd.DataFrame - dataframe with whole information about upcoming game
            - anchor: pd.DataFrame - dataframe with previous games both teams
        """

        if match_id is None and (not team1 or not team2): 
            raise Exception("If you not provide match_id you should provide: team1, team2, [ league_id or prize_pool ]")
        assert (team1 is None and team2 is None) or (team1 is not None and team2 is not None), \
            "Both team1 and team2 should be None or not None"
        assert league_id is not None or prize_pool is not None or match_id is not None, \
            'Please provide at least one argument: league_id, prize_pool, match_id'

        if team1 is not None and self.tokenizer.tokenize(team1, teams=True) == 1:
            raise exceptions.property.RadiantUntokenizedTeam(team1)

        if team2 is not None and self.tokenizer.tokenize(team2, teams=True) == 1:
            raise exceptions.property.DireUntokenizedTeam(team2)
            
        # ----------------------------------------------------------------------------------- #
        await self.log("parse prematch anchor")
        anchor = await self.prematch_anchor(team1=team1, team2=team2, match_id=match_id, league_id=league_id, prize_pool=prize_pool)
        if team1 is None and team2 is None:
            team1 = anchor['r_team_id'].values[0]
            team2 = anchor['d_team_id'].values[0]

        tokenized_team1 = self.tokenizer.tokenize(team1, teams=True)
        tokenized_team2 = self.tokenizer.tokenize(team2, teams=True)
        if tokenized_team1 == 1: raise exceptions.property.RadiantUntokenizedTeam(team1)
        if tokenized_team2 == 1: raise exceptions.property.DireUntokenizedTeam(team2)
        await self.log("prematch anchor parsed")

        # ----------------------------------------------------------------------------------- #
        await self.log("parse and transform prematch corpus")
        corpus = await self.prematch_corpus(window_size=window_size, team1=team1, team2=team2, match_id=match_id)
        corpus = self.scaler.transform(corpus, 'yeo-johnson', mode='both')
        await self.log("prematch corpus parsed and transformed")

        return team1, team2, corpus, anchor

    async def prepare_prematch(self, team1: int, team2: int, corpus: pd.DataFrame, anchor: pd.DataFrame) -> dict:
        """Prepare `corpus` and `anchor` to feed to model. 

        !!! Before we call this we should define configs, put it in `utils.base.ConfigBase._configs`
        """
        collector = PrematchTSCollector(
            tokenizer_path=self.tokenizer_path,
            y_output='crossentropy', 
            teams_reg_output=False,
            mask_type='bool', 
        )
        # ----------------------------------------------------------------------------------- #
        await self.log('collect windows')
        sample = collector.collect_windows(games=corpus, anchor=anchor, tokenize=True)
        await self.log('num of radiant games:', sample['r_window']['seq_len'])
        await self.log('num of dire games:', sample['d_window']['seq_len'])

        # ----------------------------------------------------------------------------------- #
        f_config = self._get_config('features')
        if sample['r_window']['seq_len'] < f_config['league']['window_min_size']:
            raise exceptions.property.RadiantNotEnoughGames(team1)

        if sample['d_window']['seq_len'] < f_config['league']['window_min_size']:
           raise exceptions.property.DireNotEnoughGames(team2)

        return sample

    async def collect_players_stack(self, team: int) -> list[int]:
        stack = await self.opendota_wrapper.fetch_team_stack(team)
        if len(stack) > 5: await self.log("Error in stack")
        return stack[:5]

    async def prematch_anchor(self, team1: int|None = None, team2: int|None = None, match_id: int|None = None, league_id: int|None = None, prize_pool: int|None = None):
        if match_id is not None:
            # Build anchor by match id
            anchor_match = await self.parse_match(match_id=match_id)
            if anchor_match is not None:
                # Parse opendota for a match
                anchor_match = self.opendota_parser(anchor_match)
                anchor = {
                    "match_id": [match_id],
                    "radiant_win": [anchor_match.radiant_win],
                    "r_team_id": [anchor_match.teams.radiant.id],
                    "d_team_id": [anchor_match.teams.dire.id],
                    "league_prize_pool": anchor_match.league.prize_pool
                }
                r_stack = [p.account_id for p in anchor_match.players.radiant]
                d_stack = [p.account_id for p in anchor_match.players.dire]

            else:
                # Parse Anchor from live league game
                anchor_match = await self.steam_wrapper.fetch_live_league_game(match_id)
                assert anchor_match is not None, f'{match_id=} not found'

                prize_pool = await self.get_prize_pool_by_league_id(anchor_match['league_id'])

                anchor = {
                    "match_id": [match_id],
                    "radiant_win": [-1],
                    "r_team_id": [anchor_match['radiant_team']['team_id']],
                    "d_team_id": [anchor_match['dire_team']['team_id']],
                    "league_prize_pool": prize_pool
                }
                r_stack = [p['account_id'] for p in anchor_match['players'] if p['team'] == 0]
                r_stack = [p['account_id'] for p in anchor_match['players'] if p['team'] == 1]
                
        else:
            # Build anchor from scratch
            if prize_pool is None:
                prize_pool = await self.get_prize_pool_by_league_id(league_id)

            anchor = {
                "match_id": [0],
                "radiant_win": [-1],
                "r_team_id": [team1],
                "d_team_id": [team2],
                "league_prize_pool": prize_pool,
            }
            r_stack = await self.collect_players_stack(team1)
            d_stack = await self.collect_players_stack(team2)

        anchor.update({f"{s}_account_id": [p] for s, p in zip(self.RADIANT_SIDE, r_stack)})
        anchor.update({f"{s}_account_id": [p] for s, p in zip(self.DIRE_SIDE, d_stack)})

        anchor = pd.DataFrame(anchor)
        anchor['league_prize_pool'] = anchor['league_prize_pool'].map(scalers.vectorize_prize_pool)
        return anchor

    async def prematch_corpus(self, window_size:int, team1: int, team2: int, match_id:int|None=None):
        team1_matches, team2_matches = await execute_cor(
            self.opendota_wrapper.fetch_team_games(team1), 
            self.opendota_wrapper.fetch_team_games(team2),
        )

        if match_id is not None:
            team1_matches = [match for match in team1_matches if match['match_id'] < match_id]
            team2_matches = [match for match in team2_matches if match['match_id'] < match_id]

        # ----------------------------------------------------------------------------------- #
        # optimize teams matches, remove duplicatase
        team1_matches_ids = [match['match_id'] for match in team1_matches]
        team2_matches_ids = [match['match_id'] for match in team2_matches]
        team1_matches = [m for m, mid in zip(team1_matches, team1_matches_ids) if mid not in team2_matches_ids]

        # ----------------------------------------------------------------------------------- #
        await self.log('parse team1 matches')
        team1_matches = await self.parse_team_matches(window_size, team1_matches)
        await self.log('parse team2 matches')
        team2_matches = await self.parse_team_matches(window_size, team2_matches)

        # ----------------------------------------------------------------------------------- #
        await self.log("property_parser")
        corpus = self.property_parser(team1_matches + team2_matches)
        corpus['league_prize_pool'] = corpus['league_prize_pool'].map(scalers.vectorize_prize_pool)
        # //TODO: FIXIT, this drop should be in parsing, (IN EVALUATOR)
        corpus = corpus[~(corpus['leavers'] > 0)]
        await self.log("Num of games in corpus:", len(corpus))
        
        return corpus

    async def parse_team_matches(self, window_size:int, matches: _typing.opendota.TeamMatches) -> list[_typing.property.Match]:
        team_matches, batch, ids = [], [], set()
        for idx, tmatch in enumerate(matches):
            match_id = tmatch['match_id']
            if self.tokenizer.tokenize(tmatch['opposing_team_id'], teams=True) > 1:
                # //TODO: fix this start_time crutch
                if time.time() - tmatch['start_time'] < 47_304_000:
                    if match_id not in ids:
                        batch.append(  self.parse_match(match_id=match_id) )
                        await self.log(f"{match_id=} parse {idx} game, num of parsed_games: {len(batch)}")
                    else: await self.log(f"{match_id=} has been skipped, duplicated")        
                else: await self.log(f"{match_id=} has been skipped, too old")    
            else: await self.log(f"{match_id=} has been skipped, bad opponent: {tmatch['opposing_team_id']}")

            if len(batch) == self.batch_size:
                await self.__parse_batch(batch, ids, team_matches)
                batch = []

            if len(team_matches) >= window_size:
                break
            
        if batch and len(team_matches) < window_size: 
            await self.__parse_batch(batch, ids, team_matches)

        await self.log("done :", len(team_matches))
        return team_matches
        
    async def save_match_to_db(self, match: _typing.opendota.Match):
        self.table.find_one_and_replace(
            filter={'match_id': match['match_id']}, 
            replacement=match, 
            upsert=True)

    async def get_match_from_db(self, match_id: int) -> _typing.opendota.Match | None:
        """returns `None` if not founded"""
        return self.table.find_one({'match_id': match_id})

    async def parse_match(self, match_id: int) -> _typing.opendota.Match | _typing.steam.LeagueGame:
        match = await self.get_match_from_db(match_id=match_id)
        if match is not None: return match

        match = await self.opendota_wrapper.force_fetch_game(match_id)
        if match is not None: 
            await self.save_match_to_db(match=match)
            return match

    async def __parse_batch(self, batch: list, ids: set, team_matches:list|list[_typing.opendota.Match]):
        done = await asyncio.gather(*batch, return_exceptions=True)

        od_matches: _typing.opendota.Match
        od_matches = [item for item in done if not isinstance(item, Exception)]
        errors = [item for item in done if isinstance(item, Exception)]
        await self.log(f"Num of errors in parse: {len(errors)}")
        [await self.log(item) for item in errors if isinstance(item, Exception)]

        for od_match in od_matches:
            property_match = await self.parse_od_match(od_match)
            if self.evaluator(property_match):
                team_matches.append(property_match)
                ids.add(od_match['match_id'])
            else: 
                await self.log(f"match_id={property_match.match_id} has been skipped unevaluated game")

            # if len(team_matches) >= self.window_size: break

        await self.log(f"Num of parsed games: {len(team_matches)}")

    async def parse_od_match(self, match: _typing.opendota.Match):
        try:
            property_match = self.opendota_parser(match)
            return property_match

        except exceptions.property.LeaguesJSONsNotFound as e:
            await self._scarpe_leagues()
            await self._scarpe_prize_pool(e.leagueid)

        except exceptions.property.LeaguePPNotFound as e:
            await self._scarpe_prize_pool(e.leagueid)
            
        except exceptions.property.LeagueIDNotFound as e:
            await self._scarpe_leagues()

        self.opendota_parser = parsers.OpendotaParser()
        return await self.parse_od_match(match=match)
        
    async def _scarpe_leagues(self):
        leagues = await self.opendota_wrapper.fetch_leagues()
        path = self._get_relative_path()
        path = path.parent.resolve()

        with open(path / 'scarpe/output/leagues.json', 'w', encoding='utf-8') as f:
            json.dump(leagues, f, ensure_ascii=False, indent=4)

    async def _scarpe_prize_pool(self, leagueid):
        pool = await self.steam_wrapper.fetch_tournament_prize_pool(leagueid=leagueid)
        path = self._get_relative_path()
        path = path.parent.resolve()

        with open(path / 'scarpe/output/prize_pools.json', 'r', encoding='utf-8') as f:
            prize_pools: dict[str, int] = json.load(f)

        league_id = str(pool['result']['league_id'])
        prize_pools[league_id] = pool['result']['prize_pool']
        with open(path / 'scarpe/output/prize_pools.json', 'w', encoding='utf-8') as f:
            json.dump(prize_pools, f, ensure_ascii=False, indent=4)
        
    async def get_prize_pool_by_league_id(self, league_id):
        if league_id in self.opendota_parser.prize_pools:
            prize_pool = self.opendota_parser.prize_pools[league_id]
        else:
            await self._scarpe_prize_pool(league_id)
            self.opendota_parser = parsers.OpendotaParser()
            prize_pool = self.opendota_parser.prize_pools[league_id]

        return prize_pool