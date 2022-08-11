import time
import asyncio
import datetime


import aiohttp


def currentSec() -> int:
    date = datetime.datetime.now()
    return date.second


class MonthLimit(Exception):
    'month limit is over'


class Session:
    def __init__(self, retries=3, timeout=65, verbose=False):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        self.retries = retries
        self.verbose = verbose


    async def get(self, URL: str, counter=0, json=None, status=666): 
        if counter > self.retries: return json, status

        try:
            response = await self.session.get(URL)
            status = response.status
            if status//100 == 2:
                json = await response.json()
                return json, status
            else:
                json, status = await self.get(URL=URL, counter=counter+1, json=json, status=status)
                return json, status

        # Except timeout error
        except Exception as e:
            json, status = await self.get(URL=URL, counter=counter+1, json=json, status=status)
            return json, status


    async def post(self, URL: str, counter=0, json=None, status=666): 
        if counter > self.retries: return json, status

        try:
            response = await self.session.post(URL)
            status = response.status
            if status//100 == 2:
                json = await response.json()
                return json, status
            else:
                json, status = await self.post(URL=URL, counter=counter+1, json=json, status=status)
                return json, status

        # Except timeout error
        except Exception as e:
            json, status = await self.post(URL=URL, counter=counter+1, json=json, status=status)
            return json, status
            

    async def close(self):
        try:
            await self.session.close()
        except:
            pass


    async def reset(self):
        await self.close()
        self.session = aiohttp.ClientSession(timeout=self.timeout)


class OpendotaSession(Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apiLimit(self, seconds: int, headers: str, status: int):
        if int(headers['x-rate-limit-remaining-month']) > 0:
            if int(headers['x-rate-limit-remaining-minute']) <= 0:
                if self.verbose:
                    print('minute limit is over', 'sleep {} sec'.format(60-seconds + 1), flush=True)    

                # block thread
                time.sleep(60-seconds + 1)
            else:
                if self.verbose:
                    print(status)
        else:
            raise MonthLimit

    async def get(self, URL: str, counter=0, json=None, status=666): 
        if counter > self.retries: return json, status

        try:
            current_sec = currentSec()
            response = await self.session.get(URL)
            status = response.status
            if status//100 == 2:
                json = await response.json()
                return json, status
            else:
                self.apiLimit(current_sec, response.headers, status)
                json, status = await self.get(URL=URL, counter=counter+1, json=json, status=status)
                return json, status

        # Except timeout error
        except Exception as e:
            json, status = await self.get(URL=URL, counter=counter+1, json=json, status=status)
            return json, status


    async def post(self, URL: str, counter=0, json=None, status=666): 
        if counter > self.retries: return json, status

        try:
            current_sec = currentSec()
            response = await self.session.post(URL)
            status = response.status
            if status//100 == 2:
                json = await response.json()
                return json, status
            else:
                self.apiLimit(current_sec, response.headers, status)
                json, status = await self.post(URL=URL, counter=counter+1, json=json, status=status)
                return json, status

        # Except timeout error
        except Exception as e:
            json, status = await self.post(URL=URL, counter=counter+1, json=json, status=status)
            return json, status


