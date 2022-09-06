import sys; sys.path.append("../")
import logging
import asyncio

import pymongo
from fuzzywuzzy import fuzz

import utils
from utils import _typing
from utils.development import suppress


def log(*args):
    s = ""
    for a in args:
        s += str(a)
        s += " "
    # logging.info(s)
    print(s)


class HawkParser:
    def __init__(self):
        self.steam = utils.wrappers.SteamWrapper()
        self.hawk = utils.wrappers.HawkWrapper()

        client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = client['Odds']

        self.match_to_ignore = []
        self.watch: dict[int, _typing.hawk.Match] = {}
        self.series: list[_typing.hawk.Series] = []
        self.league_games: list[_typing.steam._league_game] = []

        self.__started = False


    async def run(self):
        self.__started = True
        await asyncio.gather(self.__series(), self.__matches(), self.__watch_games())

    async def stop(self):
        self.__started = False

    async def __series(self):
        log("start __series")
        while self.__started:
            async with suppress(Exception, print_exc=True, trigger=asyncio.sleep(300)):
                # Get all current live leagues
                log("__series: Parse series")
                self.series = await self.hawk.parse_series()
                log("__series: Series has been parsed")

    async def __matches(self):
        log("start __matches")
        while self.__started:
            if self.series:
                async with suppress(Exception, print_exc=True, trigger=asyncio.sleep(10)):
                    # Get all current live games
                    tasks = []
                    for serie in self.series:
                        if serie["matches"]:
                            match = serie["matches"][-1]
                            # None - match is not over yet 
                            if match["is_radiant_won"] is None:
                                if match['id'] not in self.match_to_ignore:
                                    log(f"__matches: add {match['id']=} to parse")
                                    tasks.append( self.hawk.parse_match(match["id"]) )
                                
                    if tasks:
                        # Parse games
                        log(f"__matches: start parse")
                        done = await asyncio.gather(*tasks, return_exceptions=True)

                        log(f"__matches: parsed")
                        games: list[_typing.hawk.Match] = [item for item in done if not isinstance(item, Exception)]

                        erros = [item for item in done if isinstance(item, Exception)]
                        if erros:
                            log(f"__matches: errors while parse:")
                            [log(item) for item in erros]
                        

                        for game in games:
                            if not game["odds"]:
                                log(f"__matches: match {game['id']=} has no odds, add to ignore")
                                self.match_to_ignore.append(game["id"])

                            elif len(game["radiant_heroes"]) == 0 and len(game["dire_heroes"]) == 0:
                                if game["id"] not in self.watch:
                                    # Add just started match to observation
                                    self.watch[game["id"]] = game

                            elif len(game["radiant_heroes"]) == 5 and len(game["dire_heroes"]) == 5:
                                if game["id"] in self.watch:
                                    self.watch[game["id"]] = game
                                    self.match_to_ignore.append(game["id"])

                                else:
                                    log(f"__matches: parsed game has heroes, so just put it in db")
                                    __id = self.db['hawkRaw'].find_one_and_replace(
                                        filter={'id': game["id"]}, 
                                        replacement=game, 
                                        upsert=True,
                                    )

                                    if type(__id) is dict:
                                        __id = __id['_id']
                                        log(f"__matches: has been replacment {__id=}")
                                    else:
                                        log(f"__matches: has been inserted")
                                self.match_to_ignore.append(game["id"])
                            else:
                                self.match_to_ignore.append(game["id"])

            else:
                await asyncio.sleep(2)

    async def __watch_games(self):
        log("start __watch_games")
        while self.__started:
            if self.watch:
                async with suppress(Exception, print_exc=True, trigger=asyncio.sleep(10)):
                    founded_matches: dict[int, list[_typing.steam._league_game]]= {}
                    for partner in [0, 1, 2]:
                        log(f"__watch_games: fetch_live_games...")
                        live_league_games = await self.steam.fetch_live_games(partner=partner)
                        league_games = live_league_games["result"]["games"]
                        
                        # List with hawk matche's id to delete after the loop
                        to_del = []
                        for _id in self.watch:
                            log(f"__watch_games: watch {_id=} game")
                            hawk_match = self.watch[_id]

                            # heroes has been appeared
                            if len(hawk_match["radiant_heroes"]) == 5 and len(hawk_match["dire_heroes"]) == 5:
                                log(f"__watch_games: heroes has been appeared")
                                for book_name in hawk_match["odds"]:
                                    hawk_match["odds"][book_name][-1]["live"] = True

                                if hawk_match["is_radiant_won"] is not None:
                                    log(f"__watch_games: match has not ended, put it in bd")
                                    __id = self.db['hawkParsed'].insert_one(hawk_match)
                                    __id = str(__id.inserted_id)
                                    log(f"__watch_games: {__id=}")
                                    to_del += [_id]
                                    continue
                                
                                founded_matches[_id] = []
                                log(f"__watch_games: try to find match_id")
                                for league_game in league_games: 
                                    with suppress(Exception, print_exc=True):
                                        radiant_picks = league_game["scorebord"]["radiant"]["picks"]
                                        dire_picks = league_game["scorebord"]["dire"]["picks"]

                                        radiant_picks = set([d["hero_id"] for d in radiant_picks])
                                        dire_picks = set([d["hero_id"] for d in dire_picks])

                                        # We found this match in steam api
                                        if radiant_picks == set(hawk_match["radiant_heroes"]) and dire_picks == set(hawk_match["dire_heroes"]):

                                            games_id = [game["match_id"] for game in founded_matches[_id]]
                                            if league_game["match_id"] not in games_id:
                                                log(f"__watch_games: founded match with the same drafts, {league_game['match_id']=}")
                                                founded_matches[_id] += [league_game]

                        for _id in to_del:
                            log(f"__watch_games: match {_id=} in ignore list")
                            # Delete complete matches
                            self.match_to_ignore.append(_id)
                            del self.watch[_id]
                        await asyncio.sleep(5)

                    for _id in founded_matches:
                        with suppress(Exception, print_exc=True):
                            _founded_matches = founded_matches[_id]
                            if len(_founded_matches) > 1:
                                # Всё оч сложно мы нашли несколько матчей с такими драфтами
                                # Распарсим названия команд, сопоставим самую подходящую
                                # Я заебался комментировать это говно
                                hawk_match = self.watch[_id]
                                # query team name
                                query = hawk_match["radiant"]["name"] + " - " + hawk_match["dire"]["name"] 
                                # match_id - value
                                team_names: dict[int, int] = {}
                                for league_game in _founded_matches:
                                    value = league_game["radiant_team"]["team_name"] + " - " + league_game["dire_team"]["team_name"]
                                    team_names[league_game["match_id"]] = fuzz.ratio(query, value)
                                
                                # Айди матча с наибольшим соотвествием
                                match_id = max(team_names, key=team_names.get)
                                if team_names[match_id] > 50:
                                    # Put in bd
                                    hawk_match["match_id"] = match_id
                                    self.db['hawkParsed'].insert_one(hawk_match)

                            elif len(_founded_matches) == 1:
                                # Всё просто, мы нашли только один матч с такими драфтами
                                # он и будет считаться за правильный
                                league_game = _founded_matches[0]
                                hawk_match = self.watch[_id]
                                # Put in bd
                                hawk_match["match_id"] = league_game["match_id"]
                                self.db['hawkParsed'].insert_one(hawk_match)

                            log(f"__watch_games: match {_id=} in ignore list")
                            del self.watch[_id]
            else:
                await asyncio.sleep(2)

    async def __scarpe_league(self):
        async with suppress(Exception, print_exc=True, trigger=asyncio.sleep(10)):
            self.league_games = []
            for partner in [0, 1, 2]:
                log(f"__watch_games: fetch_live_games...")
                live_league_games = await self.steam.fetch_live_games(partner=partner)
                league_games = live_league_games["result"]["games"]

                self.league_games += league_games


if __name__ == "__main__":
    logging.basicConfig(filename="logs/odds.log", level=logging.INFO)

    log("-----------------")
    log("Start new session")
    hawk_parser = HawkParser()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(hawk_parser.run())
    loop.close()
 