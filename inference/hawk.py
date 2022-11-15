import sys
import time
import random
import logging
import asyncio
import pathlib
if __name__ == '__main__':
    SCRIPT_DIR = pathlib.Path(__file__).parent
    sys.path.append(str(SCRIPT_DIR.parent))

import pymongo
from fuzzywuzzy import fuzz

from utils import wrappers
from utils import _typing, exceptions
from utils.base import DotaconstantsBase
from utils.development import suppress

def sleeper(sleep, std):
    assert sleep > 0
    assert std > 0

    a1 = max(sleep - std, 0)
    a2 = sleep + std
    return random.randint(a1, a2)

def log(*args):
    s = ""
    for a in args:
        s += str(a)
        s += " "
    logging.info(s)
    print(s)

class HawkOddsParser(DotaconstantsBase):
    def __init__(self):
        self.steam = wrappers.SteamWrapper()
        self.hawk = wrappers.HawkWrapper()

        client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = client['Odds']

        self.series: list[_typing.hawk.Series] = []
        self.league_games: dict[int, _typing.steam.LeagueGame] = {}

        # matches in observation
        self.observe = []
        # matches to ignore
        self.match_to_ignore = []
        self.__started = False

    async def start(self):
        self.__started = True
        self.loop = asyncio.get_event_loop()
        await asyncio.gather(self.watch_series(), self.watch_matches(), self.watch_live_league_games())

    async def stop(self):
        self.__started = False

    # --------------------------------------------------------------------------------------- #
    async def watch_series(self):
        """Watch for the series in main hawk page"""
        while self.__started:
            async with suppress(Exception, print_exc=True, trigger=asyncio.sleep(sleeper(300, 60))):
                log("watch_series: Parse series")
                self.series = await self.hawk.parse_series()
                log("watch_series: Series has been parsed")

    # --------------------------------------------------------------------------------------- #
    async def watch_matches(self):
        """Watch for the related matches/games"""
        while self.__started:
            async with suppress(Exception, print_exc=True, trigger=asyncio.sleep(sleeper(5, 1))):
                # log(f"watch_matches: __parse_matches")
                matches = await self.__parse_matches()
                # log(f"watch_matches: parsed")
                for match in matches: 
                    await self.__check_and_process_match(match)

    async def __parse_matches(self) -> list[_typing.hawk.Match]:
        """Get all current live matches, matches the same as games"""
        tasks = []
        matches = [serie["matches"][-1] for serie in self.series if serie["matches"]]
        for match in matches:
            # None - match is not over yet 
            # match id not in match_to_ignore - match we did not saw before, 
            # or we still waiting fot its start
            if match["is_radiant_won"] is None and match['id'] not in self.match_to_ignore:
                tasks.append( self.hawk.parse_match(match["id"]) )
            
        if tasks:
            # Parse games
            done = await asyncio.gather(*tasks, return_exceptions=True)
            [log("Error while parse:", item) for item in done if isinstance(item, Exception)]
            return [item for item in done if not isinstance(item, Exception)]

        else:
            return [] 

    async def __check_and_process_match(self, match: _typing.hawk.Match):
        """Check for the match status and procces it

        if match has no odds add it to ignore
        if match has no heroes add it to observation
        if match has a heroes put it raw to db"""

        def check_for_odds(odds: dict[str, _typing.hawk._odd]):
            if not odds:
                return False

            for book_name in odds:
                if odds[book_name]: 
                    return True

            return False

        with suppress(Exception, print_exc=True):
            match_id = match["id"]
            r_heroes = len(match["radiant_heroes"]) 
            d_heroes = len(match["dire_heroes"])
            if not check_for_odds(match["odds"]):
                log(f"watch_matches: {match_id=} - match has no odds, add to ignore")
                self.match_to_ignore.append(match_id)

            # Not started game
            elif r_heroes == 0 and d_heroes == 0:
                if match_id not in self.observe:
                    # Add just started match to observation
                    log(f"watch_matches: {match_id=} - add just started match to observe")
                    self.observe.append(match_id)

            # Started game
            elif r_heroes == 5 and d_heroes == 5:
                self.match_to_ignore.append(match_id)
                if match_id in self.observe:
                    log(f"watch_matches: {match_id=} - heroes appeared in the observed match, remove from observe and process match")
                    self.observe.remove(match_id)
                    self.loop.create_task(self.__process_match(match))

                else:
                    log(f"watch_matches: {match_id=} - put raw match  in bd")
                    await self.__process_raw_match(match)

            else: 
                log(f"watch_matches: {match_id=} - Something strange")
                self.match_to_ignore.append(match_id)

    async def __process_raw_match(self, hawk_match: _typing.hawk.Match):
        """Put bad raw hawk match in BD"""
        id = self.db['hawkRaw'].find_one_and_replace(
            filter={'id': hawk_match['id']}, 
            replacement=hawk_match, 
            upsert=True,
        )
        if type(id) is dict:
            log(f"watch_matches: {hawk_match['id']} - has been replacment {id['_id']}")
        else:
            log(f"watch_matches: {hawk_match['id']} - has been inserted")

    async def __process_match(self, hawk_match: _typing.hawk.Match):
        """Find dota match_id, live-game odds timestamp and add to BD
        
        Exctract live-game odds timestamp -> wait a few minutes ->
        get updated hawk match -> mark live-game odds ->
        find this match in stam api -> if found add to `parsedHawk` DB else `rawHawk`
        """
        match_id = hawk_match['id']
        log(f"__process_match: {match_id} - start procces hawk match_id")
        with suppress(Exception, print_exc=True):
            # ---------------------------------------------------------------------------------- #
            # Exctract the odds
            live_game_odss: dict[str, int] = {}
            for book_name in hawk_match["odds"]:
                if hawk_match["odds"][book_name]:
                    live_game_odss[book_name] = hawk_match["odds"][book_name][-1]['created_at']

            # ---------------------------------------------------------------------------------- #
            # Wait for updates
            await asyncio.sleep(120)
            log(f"__process_match: {match_id} - sleeped")

            # ---------------------------------------------------------------------------------- #
            # get updated hawk match
            try:
                updated_hawk_match = await self.hawk.parse_match(hawk_match["id"]) 
                log(f"__process_match: {match_id} - hawk match has been updated")
            except Exception as e:
                log(f"__process_match: {match_id} - hawk match has not been updated: {e}")
            
            # ---------------------------------------------------------------------------------- #
            # Mark the odds
            new_books: list[str] = []
            for book_name in updated_hawk_match["odds"]:
                if book_name in live_game_odss:
                    # Mark new  odds as live odds 
                    for idx, live_game_odds in enumerate(updated_hawk_match["odds"][book_name]):
                        if live_game_odds['created_at'] >= live_game_odss[book_name]:
                            updated_hawk_match["odds"][book_name][idx]["live"] = True
                else:
                    # If there new bookmaker drop it latter
                    new_books.append(book_name)

            # Drop bad books 
            for book_name in new_books:
                del updated_hawk_match["odds"][book_name]

            log(f"__process_match: {match_id} - odds has been marked")

            # ---------------------------------------------------------------------------------- #
            # Find dota match id for this hawk match
            founded_games_id = self.find_match(updated_hawk_match)
            if founded_games_id:
                log(f"__process_match: {match_id} - matches founded")

            # ---------------------------------------------------------------------------------- #
            # Insert in MongoDB
            updated_hawk_match["match_id"] = founded_games_id
            self.db['hawkParsed'].insert_one(updated_hawk_match)
            log(f"__process_match: {match_id} - Parsed match inserted in DB")
            
    def find_match(self, hawk_match: _typing.hawk.Match) -> list[int]:
        """Find hawk match in valve api and retuns expected match_id's"""
        log(f"find_match: {hawk_match['id']} - start search for valve match id")

        hawk_radiant_picks = set(hawk_match["radiant_heroes"])
        hawk_dire_picks = set(hawk_match["dire_heroes"])

        founded_games_id = []
        for league_game_id in self.league_games:
            league_game = self.league_games[league_game_id]

            with suppress(Exception, print_exc=False):
                radiant_picks, dire_picks = self.exctract_drafts(league_game)

                if radiant_picks == hawk_radiant_picks and dire_picks == hawk_dire_picks:
                    # We found this match in steam api
                    if league_game['match_id'] not in founded_games_id:
                        founded_games_id.append( league_game['match_id'] )
        
        return founded_games_id

    def exctract_drafts(self, league_game: _typing.steam.LeagueGame) -> tuple[set[int], set[int]]:
        """Excract draft from league game, raise exception if it impossible"""
        def validate():
            nonlocal radiant_picks
            nonlocal dire_picks

            radiant_picks = set([p for p in radiant_picks if p in self.hawk.id_to_hero])
            dire_picks = set([p for p in dire_picks if p in self.hawk.id_to_hero])

            if len(radiant_picks) == 5 and len(dire_picks) == 5:
                return True
            else:
                return False

        with suppress(Exception, print_exc=False):
            radiant_picks = [p["hero_id"] for p in league_game["scorebord"]["radiant"]["picks"]]
            dire_picks = [p["hero_id"] for p in league_game["scorebord"]["radiant"]["picks"]]
            if validate():
                return radiant_picks, dire_picks

        with suppress(Exception, print_exc=False):
            radiant_picks = [p['hero_id'] for p in league_game['players'] if p['team'] == 0]
            dire_picks = [p['hero_id'] for p in league_game['players'] if p['team'] == 1]
            if validate():
                return radiant_picks, dire_picks

        with suppress(Exception, print_exc=False):
            radiant_picks = [p['hero_id'] for p in league_game["scorebord"]["radiant"]["players"]]
            dire_picks = [p['hero_id'] for p in league_game["scorebord"]["dire"]["players"]]
            if validate():
                return radiant_picks, dire_picks

        raise exceptions.steam.LeagueGameHasNoDrafts

    # --------------------------------------------------------------------------------------- #
    async def watch_live_league_games(self):
        """Scarpe and parse valve live league games"""
        partner = 0
        while self.__started:
            async with suppress(Exception, print_exc=False, trigger=asyncio.sleep(10)):
                league_games = await self.steam.fetch_live_league_games(partner=partner)
                for game in league_games:
                    game['last_update'] = time.time()
                    self.league_games[game['match_id']] = game
                await self.__purge_old_league()

                partner += 1
                if partner > 3:
                    partner = 0

    async def __purge_old_league(self):
        """Purge old games that haven't been updated in a long time"""
        match_id_to_del = []
        for match_id in self.league_games:
            game = self.league_games[match_id]
            if time.time() - game['last_update'] > 600: # 10 minutes without updates
                match_id_to_del.append(match_id)

        for match_id in match_id_to_del:
            del self.league_games[match_id]


if __name__ == "__main__":
    logging.basicConfig(filename="logs/odds.log", level=logging.INFO)

    hawk_parser = HawkOddsParser()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(hawk_parser.start())
    loop.close()
