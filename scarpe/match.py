"""

    This shit code migrate from old sources
    //TODO: Recode this

"""

import traceback
import aiohttp
import aiofiles
import asyncio
import datetime
import pymongo
import time


class Scarper():
    def __init__(self):
        self.AVGMMR = 6000
        self.VERBOSE = True
        self.DAYS_BEFORE = 5
        self.DAYS_BEFORE_RE = 5
        self.BATCH_SIZE = 60
        self.SLEEP_AFTER_PARSE = 20 # in minutes
        self.LIMIT = 10_000

        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=65))
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client['DotaMatches']
        self.loop = asyncio.get_event_loop()
        
        self.league_matches = []
        self.pub_matches = []
        
        self.parser_tasks_l = []
        self.parser_tasks_p = []
    
    
    def start(self):
        if self.VERBOSE: print('Start...')
            
        self.loop.run_until_complete(self.getPublic())
        self.loop.run_until_complete(self.regetPub())
        if self.VERBOSE: print('Geted pub match id')
            
        self.loop.run_until_complete(self.getLeague())
        self.loop.run_until_complete(self.regetLeague())
        if self.VERBOSE: print('Geted league match id')
            
        self.loop.run_until_complete(self.parse())
        if self.VERBOSE: print('Request to parse sended, sleep 20 minutes')
        for minute in range(self.SLEEP_AFTER_PARSE):
            time.sleep(60)
            if self.VERBOSE: print(f"{minute} minute has been slept")
            
        if self.VERBOSE: print('Starts scarpe')
        self.loop.run_until_complete(self.scarpePublic())
        self.loop.run_until_complete(self.scarpeLeague())

        self.client.close()
        if self.VERBOSE: print('Done!')
            
    def currentSec(self) -> int:
        date = datetime.datetime.now()
        return date.second

    
    def apiLimit(self, seconds: int, headers: str, status: int):
        if int(headers['x-rate-limit-remaining-month']) > 0:
            if int(headers['x-rate-limit-remaining-minute']) <= 0:
                print('minute limit is over', 'sleep {} sec'.format(60-seconds + 1), flush=True)    
                # block program, we dont wanna parse next, cause api limit
                time.sleep(60-seconds + 1)
            else:
                print(status)
        else:
            for _ in range(100):
                print('month limit is over') 

                
    async def downloadReplay(self, url: str) -> bool:
        async with aiohttp.ClientSession(timeout = aiohttp.ClientTimeout(total = 0, connect = 25)) as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    file_name = url.split('/')[-1]
                    f = await aiofiles.open(f"replays/{file_name}", mode='wb')
                    await f.write(await resp.read())
                    await f.close()
                    return True

                
    async def sessionGetPost(self, URL: str, get = True, counter = 0) -> list:
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
                self.apiLimit(current_sec, response.headers, status)
                json, status = await self.sessionGetPost(URL=URL, get=get, counter=counter+1)
                return json, status
        except Exception as e:
            json, status = await self.sessionGetPost(URL=URL, get=get, counter=counter+1)
            return json, status

        
    def exists(self, match_id: int, table: pymongo.MongoClient) -> bool:
        if table.find_one({'match_id': match_id}):
            return True
        else:
            return False
        
        
    async def getPublic(self):
        ts = time.time() - (self.DAYS_BEFORE*24*60*60)
        DATE = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        json, status = await self.sessionGetPost(
            f"""https://api.opendota.com/api/explorer?sql=
            SELECT%20*%0AFROM%20public_matches%0AWHERE%20TRUE%0AAND%20
            public_matches.start_time%20%3E%3D%20
            extract(epoch%20from%20timestamp%20%27{DATE}%27)%0AAND%20
            public_matches.AVG_MMR%20%3E%3D%20{self.AVGMMR}%0ALIMIT%20{self.LIMIT}"""
        )

        if status//100 == 2:      
            if self.VERBOSE:
                print('SQL query for public succes')      
            
            match_info = json['rows']
            table = self.db['public']
            for match in match_info:
                table.find_one_and_replace(filter = {'match_id': match['match_id']}, replacement = match, upsert = True)
            
            table = self.db['publicMatches']
            unparsed_matches = [match for match in match_info if not self.exists(match['match_id'], table)]
            match_ids = [match['match_id'] for match in unparsed_matches]
            self.pub_matches = self.pub_matches + match_ids
        else:
            if self.VERBOSE:
                print('SQL query for public IS NOT SUCCES')
                
                
    async def getLeague(self):
        ts = time.time() - (self.DAYS_BEFORE*24*60*60)
        DATE = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        json, status = await self.sessionGetPost(
            f"""https://api.opendota.com/api/explorer?sql=
            SELECT%20%0Amatches.match_id%2C%0Aleagues.name%20leaguename%2C%0Aleagues.tier%20leaguetier%2C%0Amatches.start_time%2C%0Amatches.lobby_type%2C%0Amatches.human_players%2C%0Amatches.game_mode%2C%0Amatches.radiant_team_id%2C%0Amatches.dire_team_id%2C%0Amatches.radiant_captain%2C%0Amatches.dire_captain%0A
            FROM%20matches%0AJOIN%20leagues%20using(leagueid)%0A
            WHERE%20matches.start_time%20%3E%3D%20extract(epoch%20from%20timestamp%20%27{DATE}%27)%0A
            ORDER%20BY%20matches.start_time%20DESC"""
        )
        if status//100 == 2:      
            if self.VERBOSE:
                print('SQL query for league succes')      
            
            match_info = json['rows']
            table = self.db['league']
            for match in match_info:
                table.find_one_and_replace(filter = {'match_id': match['match_id']}, replacement = match, upsert = True)
            
            table = self.db['leagueMatches']
            unparsed_matches = [match for match in match_info if not self.exists(match['match_id'], table)]
            match_ids = [match['match_id'] for match in unparsed_matches]
            self.league_matches = self.league_matches + match_ids
        else:
            if self.VERBOSE:
                print('SQL query for league IS NOT SUCCES')
                
                
    async def parse(self):
        match_ids = self.league_matches + self.pub_matches
        tasks = [self.sessionGetPost(f"https://api.opendota.com/api/request/{match_id}", get = False) for match_id in match_ids]
        if tasks:
            for batch in range(0, len(tasks), self.BATCH_SIZE):
                await asyncio.gather(*tasks[batch:batch+self.BATCH_SIZE])
            
            
    async def scarpePublic(self):
        table = self.db['publicMatches']
        api_tasks = [self.sessionGetPost(f"https://api.opendota.com/api/matches/{match_id}") for match_id in self.pub_matches if not self.exists(match_id, table)]
        for batch in range(0, len(api_tasks), self.BATCH_SIZE):
            if self.VERBOSE:
                print('batch :', batch, 'from :', len(api_tasks))

            time_start = time.time()
            # Scarp in batches, anyway we have a limits
            done = await asyncio.gather(*api_tasks[batch:batch+self.BATCH_SIZE])
            for item in done:
                json, code = item
                if code//100 == 2:
                    try: self.parser_tasks_p.append(self.downloadReplay(json['replay_url']))
                    except: pass
                    try: table.find_one_and_replace(filter = {'match_id': json['match_id']}, replacement = json, upsert = True)
                    except Exception:
                        traceback.print_exc()

            time_remaining = time.time() - time_start
            if time_remaining < 60:
                time.sleep(60 - time_remaining)
                
    async def scarpeLeague(self):
        table = self.db['leagueMatches']
        api_tasks = [self.sessionGetPost(f"https://api.opendota.com/api/matches/{match_id}") for match_id in self.league_matches if not self.exists(match_id, table)]
        for batch in range(0, len(api_tasks), self.BATCH_SIZE):
            if self.VERBOSE:
                print('batch :', batch, 'from :', len(api_tasks))

            time_start = time.time()
            # Scarp in batches, anyway we have a limits
            done = await asyncio.gather(*api_tasks[batch:batch+self.BATCH_SIZE])
            for item in done:
                json, code = item
                if code//100 == 2:
                    try: self.parser_tasks_l.append(self.downloadReplay(json['replay_url']))
                    except: pass
                    try: table.find_one_and_replace(filter = {'match_id': json['match_id']}, replacement = json, upsert = True)
                    except Exception:
                        traceback.print_exc()

            time_remaining = time.time() - time_start
            if time_remaining < 60:
                time.sleep(60 - time_remaining)
                
                
    async def regetPub(self):
        """This needs only if match
        did not have enough time to 
        parse"""
        table = self.db['publicMatches']
        unparsed_matches = table.find(
            {
                "objectives": None,
                "start_time": {
                    "$gt": int(time.time()) - 60*60*24*self.DAYS_BEFORE_RE
                }
                
            }
        )
        if unparsed_matches:
            match_ids = [match['match_id'] for match in unparsed_matches]
            self.pub_matches = self.pub_matches + match_ids
            
    async def regetLeague(self):
        """This needs only if match
        did not have enough time to 
        parse"""
        table = self.db['leagueMatches']
        unparsed_matches = table.find(
            {
                "objectives": None,
                "start_time": {
                    "$gt": int(time.time()) - 60*60*24*self.DAYS_BEFORE_RE
                }
                
            }
        )
        if unparsed_matches:
            match_ids = [match['match_id'] for match in unparsed_matches]
            self.league_matches = self.league_matches + match_ids


scarper = Scarper()
if __name__ == '__main__':
    scarper.start()