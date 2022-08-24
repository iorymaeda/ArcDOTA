import os
import time
import asyncio
import datetime
import pandas as pd

# import pymongo
import aiohttp
import aiofiles


from . import _typing, parsers, evaluator, tokenizer, scalers
from .time_series import TSCollector
from .exceptions import steam, opendota, property
from .base import ConfigBase
from .development import OpendotaSession, SessionHelper, suppress


#//TODO: fix it
os.environ['steam_api_key'] = '61886694FAAE6A81CD7E6337B6D98361'


class SteamWrapper(ConfigBase, SessionHelper):
    async def fetch_live_game(self, match_id: int) -> list[dict, bool]:
        data, status = await self.get(URL=f"http://api.steampowered.com/IDOTA2Match_570/GetLiveLeagueGames/v1?key={os.environ.get('steam_api_key')}&match_id={match_id}")
        if status//100 not in [2, 3]:
            raise steam.SteamError(status, data)

        if data['result']['games']:
            live = True
            match = data['result']['games'][0]
            return match, live
        raise steam.LiveGameNotFound(f"Game not found")
        

    async def downloadReplay(self, url: str, save_path: str) -> bool:
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
        data, status = await self.get(url + f"?{self.key}")
        if status//100 not in [2, 3]:
            raise opendota.OpendotaError(status, data)


    async def validate_match(self, match: _typing.opendota.Match):
        """Validate match as parsed or not"""
        return True if match['objectives'] else False


    async def fetch_game(self, match_id: _typing.opendota.match_id) -> _typing.opendota.Match:
        url = f'{self.BASE_URL}/matches/{match_id}'
        data, status = await self.get(url + f"?{self.key}")
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

            print('Try another one force fetch...')
            return await self.force_fetch_game(match_id, num=num-1) 

            async with suppress(opendota.OpendotaError, opendota.GameNotParsed, trigger=asyncio.sleep(30)):
                await self.parse_game(match_id=match_id)
                return await self.fetch_game(match_id=match_id)

            print('Try another one force fetch...')
            
         
    async def fetch_team_games(self, team: int) -> _typing.opendota.TeamMatches:
        url = f'{self.BASE_URL}/teams/{team}/matches/'
        games, status = await self.get(url + f"?{self.key}")
        if status//100 not in [2, 3]:
            raise opendota.OpendotaError(status, games)
        return games

    
    async def fetch_teams_meetings(self, team1: int, team2: int, limit:int=None) -> list[int]:
        if limit is None: limit = 1_000
        url = f"""{self.BASE_URL}/explorer?sql=
        SELECT%20%0Amatches.match_id%2C%0Amatches.start_time%2C%0Amatches.radiant_team_id%2C%0Amatches.dire_team_id%0A
        FROM%20matches%0AWHERE%20((RADIANT_TEAM_ID%20%3D%20{team1}%20and%20DIRE_TEAM_ID%20%3D%20{team2})%20or%20(RADIANT_TEAM_ID%20%3D%20{team2}%20and%20DIRE_TEAM_ID%20%3D%20{team1}))%0A
        ORDER%20BY%20matches.start_time%20DESC%0ALIMIT%20{200}"""
        games, status = await self.get(url + f"&{self.key}")
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
        data, status = await self.get(url + f"&{self.key}")
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
        data, status = await self.get(url + f"&{self.key}")
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
        data, status = await self.session.get(url + f"?{self.key}")
        if status//100 not in [2, 3]: raise opendota.OpendotaError(status, data)

        return data


    async def fetch_team_stack(self, team: int) -> list[int]:
        url = f'{self.BASE_URL}/teams/{team}/players'
        data, status = await self.get(url + f"?{self.key}")
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

        path = self._get_curernt_path()
        path = path.parent.resolve()
        self.opendota_parser = parsers.OpendotaParser(
            dotaconstants_path= path / 'scarpe/dotaconstants',
            leagues_path=       path / 'scarpe/output/leagues.json', 
            prize_pools_path=   path / 'scarpe/output/prize_pools.json')
        self.property_parser = parsers.PropertyParser()
        self.evaluator = evaluator.Evaluator()
        self.scaler = scalers.DotaScaler(path=path / "inference/files/scaler_league.pkl")

        # //TODO: fix tokenizer, put it together with a model
        self.tokenizer = tokenizer.Tokenizer(path=path / "inference/files/tokenizer_league.pkl")
        # //TODO: mask_type, y_output and anothers hyper-parameters put to configs
        self.collector = TSCollector(
            mask_type='bool', 
            y_output='crossentropy', 
            teams_reg_output=False,
            tokenizer_path=path / "inference/files/tokenizer_league.pkl",
            )

        self.batch_size = 60

    async def collect_players_stack(self, team: int) -> list[int]:
        stack = await self.opendota_wrapper.fetch_team_stack(team)
        if len(stack) > 5: print("Error in stack")
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
        print("prematch_corpus parsed")
        anchor = await self.prematch_anchor(team1=team1, team2=team2)
        print("prematch_anchor parsed")

        corpus = self.scaler.transform(corpus, 'yeo-johnson', mode='both')
        print('collect windows')
        sample = self.collector.collect_windows(
            games=corpus, anchor=anchor, tokenize=True, 
        )
        print('collected')
        print(sample['r_window']['seq_len'])
        print(sample['d_window']['seq_len'])

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
        print('parse team1 matches')
        team1_matches = await self.parse_team_matches(team1_matches)
        print('parse team2 matches')
        team2_matches = await self.parse_team_matches(team2_matches)

        print("property_parser")
        corpus = self.property_parser(team1_matches + team2_matches)
        

        print(len(corpus))
        # //TODO: FIXIT, this drop should be in parsing, (IN EVALUATOR)
        corpus = corpus[~(corpus['leavers'] > 0)]
        print(len(corpus))
        return corpus


    async def parse_team_matches(self, matches: _typing.opendota.TeamMatches) -> list[_typing.property.Match]:
        """Synchronous parse //TODO:improve it"""
        window_size = self._get_config('features')['league']['window_size']

        ids = set()
        batch = []
        team_matches = []   
        for idx, tmatch in enumerate(matches):

            match_id = tmatch['match_id']
            print(f"parse {idx} game")
            if (self.tokenizer.tokenize(tmatch['opposing_team_id'], teams=True) > 1 and
                match_id not in ids and time.time() - tmatch['start_time'] < 47_304_000):
                # //TODO: fix this start_time crutch
                batch.append(self.opendota_wrapper.force_fetch_game(match_id))

            if len(batch) == self.batch_size:
                done = await asyncio.gather(*batch, return_exceptions=True)

                batch = []
                matches = [item for item in done if not isinstance(item, Exception)]
                [print(item) for item in done if isinstance(item, Exception)]


                for od_match in matches:
                    property_match = self.opendota_parser(od_match)
                    if self.evaluator(property_match):
                        team_matches.append(property_match)
                        ids.add(od_match['match_id'])

                    else: print(f"Skipped unevaluated")

                    if len(team_matches) >= window_size: break
                if len(team_matches) >= window_size: break


            
        print("done :", len(team_matches))
        return team_matches
        
