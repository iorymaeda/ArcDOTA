import os
import time
import json
import asyncio
import datetime
from typing import Literal

# import pymongo
import aiohttp
import aiofiles
import pandas as pd
from lxml import html

from . import _typing, parsers, evaluator, tokenizer, scalers
from .time_series import TSCollector
from .exceptions import steam, opendota, property
from .base import ConfigBase, DotaconstantsBase
from .development import OpendotaSession, SessionHelper, suppress


class HawkWrapper(SessionHelper, DotaconstantsBase):
    BASE_URL = "https://hawk.live"

    def __init__(self):
        super().__init__()    
        self.npc_to_id, self.id_to_hero = self._load_heroes()

    async def fetch_match(self, match_id: int) -> aiohttp.client.ClientResponse:
        url = f'{self.BASE_URL}/matches/{match_id}'
        response = await self.get(url, raw=True)
        if response.status//100 not in [2, 3]:
            raise Exception('Cant get hawk match')
        return response

    async def fetch_main_page(self) -> aiohttp.client.ClientResponse:
        url = f'{self.BASE_URL}/'
        response = await self.get(url, raw=True)
        if response.status//100 not in [2, 3]:
            raise Exception('Cant get hawk main page')
        return response

    async def parse_match(self, match_id: int) -> _typing.hawk.Match:
        response = await self.fetch_match(match_id=match_id)

        tree: html.HtmlElement = html.fromstring(await response.text())
        data_elems: list[html.HtmlElement] = tree.xpath('//*[@id="app"]')

        if len(data_elems) > 1: raise Exception('Unexcepted hawk html tree while fetch match')

        data = data_elems[0]
        data = dict(data.attrib)
        data = json.loads(data['data-page'])

        is_team1_radiant = data['props']['match']['is_team1_radiant']

        radiant_heroes = [ self.npc_to_id[hero['hero']['code_name']] for hero in data['props']['match']['picks'] if hero['is_radiant'] ]
        dire_heroes = [ self.npc_to_id[hero['hero']['code_name']] for hero in data['props']['match']['picks'] if not hero['is_radiant'] ]
        radiant_team = data['props']['match']['team1']
        dire_team = data['props']['match']['team2']
        is_radiant_won = data['props']['match']['is_radiant_won']

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
                        'heroes': False,
                        'created_at': created_at
                    })
                    
            match_odds_info_array[odds['odds_provider_code_name']] = _odds

        if not is_team1_radiant: 
            radiant_heroes, dire_heroes = dire_heroes, radiant_heroes
            radiant_team, dire_team = dire_team, radiant_team

        return {
            "id": match_id,
            'match_id': None,
            "is_radiant_won": is_radiant_won,
            "radiant_team": radiant_team,
            "dire_team": dire_team,
            "radiant_heroes": radiant_heroes,
            "dire_heroes": dire_heroes,
            "odds": match_odds_info_array
        }

    async def parse_series(self) -> list[_typing.hawk.Series]:
        response = await self.fetch_main_page()

        tree: html.HtmlElement = html.fromstring(await response.text())
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


    async def fetch_live_game(self, match_id: int) -> _typing.steam._league_game:
        url = f"{self.BASE_URL}/IDOTA2Match_570/GetLiveLeagueGames/v1?{match_id=}"
        url = url + f"&{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise steam.SteamError(status, data)

        data: _typing.steam.GetLiveLeagueGames
        if data['result']['games']:
            return data['result']['games'][0]
        raise steam.LiveGameNotFound(f"Game not found")
        

    async def fetch_live_games(self, partner:Literal[0, 1, 2, 3]=0) -> _typing.steam.GetLiveLeagueGames:
        url = f"{self.BASE_URL}/IDOTA2Match_570/GetLiveLeagueGames/v1?{partner=}"
        url = url + f"&{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise steam.SteamError(status, data)
        return data


    async def fetch_tournament_prize_pool(self, leagueid:int) -> _typing.steam.GetTournamentPrizePool:
        url = f"http://api.steampowered.com/IEconDOTA2_570/GetTournamentPrizePool/v1?{leagueid=}"
        url = url + f"&{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise steam.PrizePoolNotFound(status, leagueid)

        if data['result']['league_id'] == leagueid: return data
        raise steam.PrizePoolNotFound(status, leagueid)


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
            raise opendota.OpendotaError(status, data)


    async def validate_match(self, match: _typing.opendota.Match):
        """Validate match as parsed or not"""
        return True if match['objectives'] else False


    async def fetch_game(self, match_id: _typing.opendota.match_id) -> _typing.opendota.Match:
        url = f'{self.BASE_URL}/matches/{match_id}'
        url = url + f"?{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise opendota.OpendotaError(status, data)

        if not await self.validate_match(match=data):
            # If the game is not parsed and has passed a lot of time it will not be parsed
            # 86400 is one day
            if data['leagueid'] == 0:
                # Public match, here are no auto parse
                if data['start_time'] > 86400 * 8:
                    raise opendota.ParsingNotPossible
            else:
                # League match, here are auto parse,
                # So if game after one day still not parsed - broken
                if data['start_time'] > 86400:
                    raise opendota.ParsingNotPossible

            raise opendota.GameNotParsed
        return data


    async def force_fetch_game(self, match_id: _typing.opendota.match_id, num=20) -> _typing.opendota.Match:
        """Parse and fetch game"""
        num-= 1
        if num > 0:
            try:
                game = await self.fetch_game(match_id=match_id)
                return game
            except opendota.GameNotParsed:
                await self.parse_game(match_id=match_id)
                await asyncio.sleep(30)
                print('GameNotParsed: Try another one force fetch...')
                return await self.force_fetch_game(match_id, num=num) 
            
            except opendota.OpendotaError:
                await asyncio.sleep(4)
                print('OpendotaError: Try another one force fetch...')
                return await self.force_fetch_game(match_id, num=num)
            
         
    async def fetch_team_games(self, team: int) -> _typing.opendota.TeamMatches:
        url = f'{self.BASE_URL}/teams/{team}/matches/'
        url = url + f"?{self.key}" if self.key else url
        games, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise opendota.OpendotaError(status, games)
        return games

    
    async def fetch_teams_meetings(self, team1: int, team2: int, limit:int=None) -> list[int]:
        if limit is None: limit = 1_000
        url = f"""{self.BASE_URL}/explorer?sql=
        SELECT%20%0Amatches.match_id%2C%0Amatches.start_time%2C%0Amatches.radiant_team_id%2C%0Amatches.dire_team_id%0A
        FROM%20matches%0AWHERE%20((RADIANT_TEAM_ID%20%3D%20{team1}%20and%20DIRE_TEAM_ID%20%3D%20{team2})%20or%20(RADIANT_TEAM_ID%20%3D%20{team2}%20and%20DIRE_TEAM_ID%20%3D%20{team1}))%0A
        ORDER%20BY%20matches.start_time%20DESC%0ALIMIT%20{200}"""
        url = url + f"?{self.key}" if self.key else url
        games, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise opendota.OpendotaError(status, games)

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
        url = url + f"?{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise opendota.OpendotaError(status, data)
        return data['rows']


    async def fetch_league_games(self, start_time: int, limit: int = None) -> _typing.opendota.SQLLeagueMatches:
        if limit is None: limit = 100_000
        date = datetime.datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d")
        url = f"""{self.BASE_URL}/explorer?sql=
            SELECT%20%0Amatches.match_id%2C%0Aleagues.name%20leaguename%2C%0Aleagues.tier%20leaguetier%2C%0Amatches.start_time%2C%0Amatches.lobby_type%2C%0Amatches.human_players%2C%0Amatches.game_mode%2C%0Amatches.radiant_team_id%2C%0Amatches.dire_team_id%2C%0Amatches.radiant_captain%2C%0Amatches.dire_captain%0A
            FROM%20matches%0AJOIN%20leagues%20using(leagueid)%0A
            WHERE%20matches.start_time%20%3E%3D%20extract(epoch%20from%20timestamp%20%27{date}%27)%0A
            ORDER%20BY%20matches.start_time%20DESC%0ALIMIT%20{limit}"""
        url = url + f"?{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise opendota.OpendotaError(status, data)
        return data['rows']


    async def scarpe_games(self, match_ids: list[_typing.opendota.match_id], batch_size=60):
        raise NotImplementedError
        games = []
        tasks = [self.force_fetch_game(match_id) for match_id in match_ids]
        for batch in range(0, len(tasks), batch_size):
            done = await asyncio.gather(*tasks[batch:batch+batch_size])


    async def fetch_leagues(self) -> list[_typing.opendota.League]:
        url = f"{self.BASE_URL}/leagues/"
        url = url + f"?{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]: raise opendota.OpendotaError(status, data)

        return data


    async def fetch_team_stack(self, team: int) -> list[int]:
        url = f'{self.BASE_URL}/teams/{team}/players'
        url = url + f"?{self.key}" if self.key else url
        data, status = await self.get(url)
        if status//100 not in [2, 3]:
            raise opendota.OpendotaError(status, data)
        
        data: list[_typing.opendota.TeamPlayers]
        return [player['account_id'] for player in data if player['is_current_team_member']]


class PropertyWrapper(ConfigBase):
    """Scarpe, parse and prepare data to feed to model

    Be careful: ConfigBase after load model in inference will output config from train phase"""

    def __init__(self):
        self.opendota_wrapper = OpendotaWrapper()
        self.steam_wrapper = SteamWrapper()

        path = self._get_relative_path()
        path = path.parent.resolve()
        self.opendota_parser = parsers.OpendotaParser()
        self.property_parser = parsers.PropertyParser()
        self.evaluator = evaluator.Evaluator()
        self.scaler = scalers.DotaScaler(path=path / "inference/files/scaler_league.pkl")

        # //TODO: fix tokenizer, put it together with a model
        self.tokenizer = tokenizer.Tokenizer(path=path / "inference/files/tokenizer_league.pkl")
        # //TODO: put to configs mask_type, y_output and anothers hyper-parameters 
        self.collector = TSCollector(
            tokenizer_path=path / "inference/files/tokenizer_league.pkl",
            y_output='crossentropy', 
            teams_reg_output=False,
            mask_type='bool', 
        )
        self.batch_size = 60

    async def log(self, *args):
        message = ''.join(map(str, args))
        print(message)

    async def collect_players_stack(self, team: int) -> list[int]:
        stack = await self.opendota_wrapper.fetch_team_stack(team)
        if len(stack) > 5: await self.log("Error in stack")
        return stack[:5]

    async def prematch(self, team1: int, team2: int, match_id:int=None) -> dict:
        assert team1 != 0 and team2 != 0
        
        tokenized_team1 = self.tokenizer.tokenize(team1, teams=True)
        tokenized_team2 = self.tokenizer.tokenize(team2, teams=True)
        if tokenized_team1 == 1: raise property.RadiantUntokenizedTeam(team1)
        if tokenized_team2 == 1: raise property.DireUntokenizedTeam(team2)

        # Do this to reopen client session inside async function
        # This is necessary when we use Gunicorn and aiohttp session
        await self.steam_wrapper.reset()
        await self.opendota_wrapper.reset()

        corpus = await self.prematch_corpus(team1=team1, team2=team2, match_id=match_id)
        await self.log("prematch_corpus parsed")
        anchor = await self.prematch_anchor(team1=team1, team2=team2)
        await self.log("prematch_anchor parsed")

        corpus = self.scaler.transform(corpus, 'yeo-johnson', mode='both')
        await self.log('collect windows')
        sample = self.collector.collect_windows(
            games=corpus, anchor=anchor, tokenize=True, 
        )
        await self.log('collected')
        await self.log(sample['r_window']['seq_len'])
        await self.log(sample['d_window']['seq_len'])

        f_config = self._get_config('features')
        if sample['r_window']['seq_len'] < f_config['league']['window_min_size']:
            raise property.RadiantNotEnoughGames(team1)

        if sample['d_window']['seq_len'] < f_config['league']['window_min_size']:
           raise property.DireNotEnoughGames(team2)

        await self.steam_wrapper.close()
        await self.opendota_wrapper.close()
        return sample

    async def prematch_anchor(self, team1: int, team2: int, match_id:int|None=None):
        if match_id is not None:
            pass

        r_stack = await self.collect_players_stack(team1)
        d_stack = await self.collect_players_stack(team2)
        anchor = {
            "match_id": [0],
            "radiant_win": [-1],
            "r_team_id": [team1],
            "d_team_id": [team2],
        }
        anchor.update({f"{s}_account_id": [p] for s, p in zip(self.RADIANT_SIDE, r_stack)})
        anchor.update({f"{s}_account_id": [p] for s, p in zip(self.DIRE_SIDE, d_stack)})
        anchor = pd.DataFrame(anchor)
        return anchor

    async def prematch_corpus(self, team1: int, team2: int, match_id:int|None=None):
        team1_matches = await self.opendota_wrapper.fetch_team_games(team1)
        team2_matches = await self.opendota_wrapper.fetch_team_games(team2)

        if match_id is not None:
            team1_matches = [match for match in team1_matches if match['match_id'] < match_id]
            team2_matches = [match for match in team2_matches if match['match_id'] < match_id]

        # //TODO: optimize teams matches, remove duplicatase
        await self.log('parse team1 matches')
        team1_matches = await self.parse_team_matches(team1_matches)
        await self.log('parse team2 matches')
        team2_matches = await self.parse_team_matches(team2_matches)

        await self.log("property_parser")
        corpus = self.property_parser(team1_matches + team2_matches)
        

        await self.log(len(corpus))
        # //TODO: FIXIT, this drop should be in parsing, (IN EVALUATOR)
        corpus = corpus[~(corpus['leavers'] > 0)]
        await self.log(len(corpus))
        return corpus

    async def parse_team_matches(self, matches: _typing.opendota.TeamMatches) -> list[_typing.property.Match]:
        self.window_size = self._get_config('features')['league']['window_size']

        ids = set()
        batch = []
        team_matches = []   
        for idx, tmatch in enumerate(matches):
            match_id = tmatch['match_id']
            await self.log(f"parse {idx} game, {match_id=}")

            # //TODO: fix this start_time crutch
            if self.tokenizer.tokenize(tmatch['opposing_team_id'], teams=True) > 1:
                if time.time() - tmatch['start_time'] < 47_304_000:
                    if match_id not in ids:
                        batch.append(self.opendota_wrapper.force_fetch_game(match_id))
                    else: await self.log(f"{match_id=} skipped, duplicated")        
                else: await self.log(f"{match_id=} skipped, too old")    
            else: await self.log(f"{match_id=} skipped, bad opponent: {tmatch['opposing_team_id']}")


            if len(batch) == self.batch_size:
                await self.__parse_batch(batch, ids, team_matches)
                batch = []

            if len(team_matches) >= self.window_size:
                break
        

        if batch and len(team_matches) < self.window_size: 
            await self.__parse_batch(batch, ids, team_matches)

        await self.log("done :", len(team_matches))
        return team_matches
    

    async def __parse_batch(self, batch: list, ids: set, team_matches:list|list[_typing.opendota.Match]):
        done = await asyncio.gather(*batch, return_exceptions=True)

        od_matches: _typing.opendota.Match
        od_matches = [item for item in done if not isinstance(item, Exception)]
        [await self.log(item) for item in done if isinstance(item, Exception)]

        for od_match in od_matches:
            property_match = await self.parse_od_match(od_match)
            if self.evaluator(property_match):
                team_matches.append(property_match)
                ids.add(od_match['match_id'])

            else: await self.log(f"Skipped unevaluated: {property_match.match_id}")

            if len(team_matches) >= self.window_size: break

    async def parse_od_match(self, match: _typing.opendota.Match):
        try:
            property_match = self.opendota_parser(match)
            return property_match

        except property.LeaguesJSONsNotFound as e:
            await self._scarpe_leagues()
            await self._scarpe_prize_pool(e.leagueid)

        except property.LeaguePPNotFound as e:
            await self._scarpe_prize_pool(e.leagueid)
            
        except property.LeagueIDNotFound:
            await self._scarpe_leagues()
        
        self.opendota_parser = parsers.OpendotaParser()
        await self.parse_od_match(match=match)
        
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
        
