"""Stranger thins to do develompent stuff"""
import time
import asyncio
import datetime
import traceback

import aiohttp

from .exceptions import opendota


class suppress:
    """Context manager to suppress specified exceptions

    After the exception is suppressed, execution proceeds with the next
    statement following the with statement.

        with suppress(FileNotFoundError):
            os.remove(somefile)
        # Execution still resumes here even if the file was already removed

        if you need to trigger function `foo` at the end:
        with suppress(FileNotFoundError, triger=bar):
             os.remove(somefile)

        if you need to trigger async function `bar` at the Exception:
        async with suppress(FileNotFoundError, on_exc=bar(args, kwargs)):
             os.remove(somefile)

    """
    def __init__(self, *exceptions, print_exc=False, trigger=None, on_exc=None):
        self.exceptions = exceptions
        self.print_exc = print_exc
        self.trigger = trigger
        self.on_exc = on_exc
        
    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        if exctype is not None:
            # Trigger function at the exception
            self.on_exc() if self.on_exc else None
            if self.print_exc: traceback.print_exc() 
        # Trigger function at the exit
        self.trigger() if self.trigger else None
        # Return True for working well or False to raise exception
        return exctype is not None and issubclass(exctype, self.exceptions)

    async def __aenter__(self):
        pass

    async def __aexit__(self, exctype, excinst, exctb):
        if exctype is not None:
            # Trigger function at the exception
            await self.on_exc if self.on_exc else None
            if self.print_exc: traceback.print_exc() 
        # Trigger function at the exit
        await self.trigger if self.trigger else None
        # Return True for working well or False to raise exception
        return exctype is not None and issubclass(exctype, self.exceptions)


class SessionHelper:
    def __init__(self):
        self.timeout = aiohttp.ClientTimeout(total=65)
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        self.verbose = True

    @property
    def _current_sec(self) -> int:
        date = datetime.datetime.now()
        return date.second
                

    async def _request(self, method, URL: str, raw=False, counter=0, json=None, status=666) -> aiohttp.client.ClientResponse | tuple[dict, int]:
        if counter > 3: return json, status
        try:
            sec = self._current_sec
            response = await method(URL)
            status = response.status
            if status//100 in [2, 3]:
                if raw:
                    return response
                else:
                    json = await response.json()
                    return json, status
            else:
                if await self.on_bad_status(sec=sec, headers=dict(response.headers), status=status):
                    counter+=1
                return await self._request(method=method, URL=URL, counter=counter, json=json, status=status, raw=raw)

        except (aiohttp.ServerTimeoutError, asyncio.exceptions.TimeoutError):
            return await self._request(method=method, URL=URL, counter=counter+1, json=json, status=status, raw=raw)

    async def on_bad_status(self, *args, **kwargs):
        """Return `True` if need to increase counter"""
        return True

    async def get(self, URL: str, raw=False) -> aiohttp.client.ClientResponse | tuple[dict, int]: 
        return await self._request(self.session.get, URL=URL, raw=raw)

    async def post(self, URL: str, raw=False) -> aiohttp.client.ClientResponse | tuple[dict, int]: 
        return await self._request(self.session.post, URL=URL, raw=raw)
            
    async def head(self, URL: str, raw=False) -> aiohttp.client.ClientResponse | tuple[dict, int]: 
        return await self._request(self.session.head, URL=URL, raw=raw)

    async def delete(self, URL: str, raw=False) -> aiohttp.client.ClientResponse | tuple[dict, int]: 
        return await self._request(self.session.delete, URL=URL, raw=raw)

    async def close(self):
        async with suppress(Exception):
            await self.session.close()

    async def reset(self):
        await self.close()
        self.session = aiohttp.ClientSession(timeout=self.timeout)


class OpendotaSession(SessionHelper):
    def __init__(self):
        super().__init__()
    
    async def opendota_api_limit(self, sec:int , headers: dict[str, str], *args, **kwargss) -> bool | None:
        """Return `True` if this is a opendota time limit and we must try again"""
        if not isinstance(headers, dict): return True

        headers = {k.lower():v for k, v in headers.items()}
        if ("x-rate-limit-remaining-month" in headers and 
            "x-rate-limit-remaining-minute" in headers):
            if int(headers['x-rate-limit-remaining-month']) <= 0:
                raise opendota.OpendotaMonthLimit

            if int(headers['x-rate-limit-remaining-minute']) <= 0:   
                if self.verbose: print(f"Opendota API rate limit")
                # If a new minute has been begun after request - limit has skiped and
                # `_current_sec` will be less than `sec`
                if self._current_sec > sec: 
                    if self.verbose: print(f'sleep: {60-self._current_sec + 1} sec')
                    time.sleep(60-self._current_sec + 1)
                    # await asyncio.sleep(60-sec + 1)
                return False
        return False
               
    async def on_bad_status(self, *args, **kwargs):
        return await self.opendota_api_limit(*args, **kwargs)

