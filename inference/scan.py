"""This script/bot scan for live games and make prediction"""

import sys
import time
import logging
import asyncio
import pathlib
if __name__ == '__main__':
    SCRIPT_DIR = pathlib.Path(__file__).parent
    sys.path.append(str(SCRIPT_DIR.parent))


import aiosqlite

from utils import wrappers, _typing
from utils.base import DotaconstantsBase
from utils.development import suppress, SessionHelper


def log(*args):
    s = ""
    for a in args:
        s += str(a)
        s += " "
    logging.info(s)
    print(s)

class ScanBot(DotaconstantsBase):
    def __init__(self):
        self.npc_to_id, self.id_to_hero = self._load_heroes()
        self.loop = asyncio.get_event_loop()
        self.steam = wrappers.SteamWrapper()
        self.session = SessionHelper()
        self.games_cache = {}

    def run(self):
        self.loop.run_until_complete(self.start())
        self.loop.close()

    async def start(self):
        partner = 0
        await self.init_db()
        await self.wait_for_rest_api()

        while True:
            await self.scan_games(partner=partner)
            if partner > 3: partner = 0
            partner += 1
            
    async def wait_for_rest_api(self):
        while True:
            async with suppress(Exception, print_exc=False):
                response = await self.session.get(f'http://localhost:8100/status', raw=True)
                if response.status//100 in [2, 3]:
                    return

            log("Wait for localhost:8100...")
            await asyncio.sleep(5)

    async def scan_games(self, partner: int):
        async with suppress(Exception, print_exc=False, trigger=asyncio.sleep(30)):
            for game in await self.steam.fetch_live_league_games(partner=partner):
                game['last_update'] = time.time()
                async with suppress(Exception, print_exc=True):
                    if self.has_drafts(game):
                        pred = await self.predict_drafts(game)
                        if pred is not None:
                            await self.put_match_info(game)
                            await self.put_draft_pred(game["match_id"], pred)

                    elif game['match_id'] not in self.games_cache:
                        pred = await self.predict_prematch(game)
                        if pred is not None:
                            await self.put_match_info(game)
                            await self.put_prematch_pred(game["match_id"], pred)

                self.games_cache[game['match_id']] = game

            await self.purge_old_league()
    
    async def predict_prematch(self, league_game: _typing.steam.LeagueGame) -> _typing.api.Prematch:
        try:
            league_game["radiant_series_wins"]
            team1 = league_game["radiant_team"]["team_id"]
            team2 = league_game["dire_team"]["team_id"]
            league_id = league_game["league_id"]
        except Exception:
            return 

        output, status = await self.session.get(f'http://localhost:8100/predict/prematch?key=A626C899DE9ABAB14D4EC3312C363&team1={team1}&team2={team2}&league_id={league_id}')
        if status//100 not in [2, 3]:
            log(f"{team1} vs {team2} has not predicted")
        else:
            return output
            
    async def predict_drafts(self, league_game: _typing.steam.LeagueGame):
        async with suppress(Exception):
            raise NotImplementedError

    async def purge_old_league(self):
        """Purge old games that haven't been updated in a long time"""
        match_id_to_del = []
        for match_id in self.games_cache:
            game = self.games_cache[match_id]
            if time.time() - game['last_update'] > 12000: # 200 minutes without updates
                match_id_to_del.append(match_id)

        for match_id in match_id_to_del:
            del self.games_cache[match_id]

    async def init_db(self):
        async with aiosqlite.connect(SCRIPT_DIR / 'DB/Predictions.db') as db:
            await db.execute("""CREATE TABLE IF NOT EXISTS SCANED_GAMES 
            (match_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            radiant_win BOOL,
            pred_prematch REAL,
            pred_drafts REAL,

            radiant_name VARCHAR(255),
            radiant_team_id INTEGER,
            radiant_series_wins INTEGER,

            dire_name VARCHAR(255),
            dire_team_id INTEGER,
            dire_series_wins INTEGER,
            
            model_tag TEXT,
            league_id INTEGER,
            founded_time INTEGER);""")
            await db.commit()

    async def match_in_db(self, match_id: int) -> bool:
        async with aiosqlite.connect(SCRIPT_DIR / 'DB/Predictions.db') as db:
            sql = f"SELECT * FROM SCANED_GAMES WHERE match_id = {match_id}"
            cursor = await db.execute(sql)
            values = await cursor.fetchone()
            return True if values else False

    async def put_match_info(self, league_game: _typing.steam.LeagueGame):
        if not await self.match_in_db(league_game['match_id']):
            async with aiosqlite.connect(SCRIPT_DIR / 'DB/Predictions.db') as db:         
                sql = f"""
                INSERT INTO SCANED_GAMES 
                
                (match_id, 
                radiant_team_id, radiant_name, radiant_series_wins,
                dire_team_id, dire_name, dire_series_wins,
                league_id, founded_time) 

                VALUES ({league_game['match_id']}
                {league_game['radiant_team']['team_id']}, '{league_game['radiant_team']['team_name']}', '{league_game['radiant_series_wins']}',
                {league_game['radiant_team']['team_id']}, '{league_game['radiant_team']['team_name']}', '{league_game['dire_series_wins']}',
                {league_game['league_id']}, {int(time.time())}
                )"""
                await db.execute(sql)
                await db.commit()

    async def put_prematch_pred(self, match_id: int, pred:_typing.api.Prematch):
        if await self.match_in_db(match_id):
            async with aiosqlite.connect(SCRIPT_DIR / 'DB/Predictions.db') as db:         
                sql = f"""
                UPDATE SCANED_GAMES SET pred_prematch = {pred.outcome}, model_tag = {pred.model_tag}
                WHERE match_id = {match_id}"""
                await db.execute(sql)
                await db.commit()
        else:
            raise Exception("Tried to put pred to match that did not exists in db")

    async def put_draft_pred(self, match_id: int, pred: None):
        raise NotImplementedError

        if await self.match_in_db(match_id):
            async with aiosqlite.connect(SCRIPT_DIR / 'DB/Predictions.db') as db:         
                sql = f"""
                UPDATE SCANED_GAMES SET pred_drafts = {pred}, model_tag = {pred.model_tag}
                WHERE match_id = {match_id}"""
                await db.execute(sql)
                await db.commit()
        else:
            raise Exception("Tried to put pred to match that did not exists in db")

    def has_drafts(self, league_game: _typing.steam.LeagueGame) -> bool:
        """Return true if games already with heroes"""
        def validate():
            nonlocal radiant_picks
            nonlocal dire_picks

            radiant_picks = set([p for p in radiant_picks if p in self.id_to_hero])
            dire_picks = set([p for p in dire_picks if p in self.id_to_hero])

            if len(radiant_picks) == 5 and len(dire_picks) == 5:
                return True
            else:
                return False

        with suppress(Exception, print_exc=False):
            radiant_picks = [p["hero_id"] for p in league_game["scorebord"]["radiant"]["picks"]]
            dire_picks = [p["hero_id"] for p in league_game["scorebord"]["radiant"]["picks"]]
            if validate():
                return True

        with suppress(Exception, print_exc=False):
            radiant_picks = [p['hero_id'] for p in league_game['players'] if p['team'] == 0]
            dire_picks = [p['hero_id'] for p in league_game['players'] if p['team'] == 1]
            if validate():
                return True

        with suppress(Exception, print_exc=False):
            radiant_picks = [p['hero_id'] for p in league_game["scorebord"]["radiant"]["players"]]
            dire_picks = [p['hero_id'] for p in league_game["scorebord"]["dire"]["players"]]
            if validate():
                return True

        return False

if __name__ == "__main__":
    logging.basicConfig(filename="logs/live_predictor.log", level=logging.INFO)

    bot = ScanBot()
    bot.run()