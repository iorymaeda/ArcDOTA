"""Stranger thigs to do develompent stuff"""

import asyncio
import inspect
import argparse
import datetime
import traceback

import discord
import aiohttp
import aiosqlite
import pandas as pd

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
                    
                try:
                    json = await response.json()
                except Exception:
                    pass

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


class WaitSession(SessionHelper):
    def __init__(self):
        super().__init__()

        # if True we wait until it end and do not send requests
        self.__api_limit = False

    async def wait_on_rate_limit(self):
        while self.__api_limit:
            await asyncio.sleep(0.1)

    async def get(self, *args, **kwargs): 
        await self.wait_on_rate_limit()
        return await super().get(*args, **kwargs)

    async def post(self, *args, **kwargs): 
        await self.wait_on_rate_limit()
        return await super().post(*args, **kwargs)

    async def head(self, *args, **kwargs): 
        await self.wait_on_rate_limit()
        return await super().head(*args, **kwargs)

    async def delete(self, *args, **kwargs): 
        await self.wait_on_rate_limit()
        return await super().delete(*args, **kwargs)


class OpendotaSession(WaitSession):
    def __init__(self):
        super().__init__()

    async def api_limit(self, sec:int , headers: dict[str, str], *args, **kwargs) -> bool:
        """Return `True` if this is a error and we must try again

        Opendota limit resets every minute, if a new minute has been begun after request -
        limit has skiped and `_current_sec` will be less than `sec`
        
        `sec` is second before request"""
        if not isinstance(headers, dict): return True

        headers = {k.lower():v for k, v in headers.items()}
        if ("x-rate-limit-remaining-month" in headers and 
            "x-rate-limit-remaining-minute" in headers):
            if int(headers['x-rate-limit-remaining-month']) <= 0:
                raise opendota.OpendotaMonthLimit

            if int(headers['x-rate-limit-remaining-minute']) <= 0:   
                if self.verbose: print(f"Opendota API rate limit")

                if self._current_sec > sec: 
                    if self.verbose: print(f'sleep: {60-self._current_sec + 1.} sec')

                    super().__api_limit = True
                    await asyncio.sleep(60-self._current_sec + 1.)
                    super().__api_limit = False
        return False
               
    async def on_bad_status(self, *args, **kwargs):
        return await self.api_limit(*args, **kwargs)


class GithubSession(WaitSession):
    def __init__(self):
        super().__init__()

    async def api_limit(self, sec:int , headers: dict[str, str], *args, **kwargs) -> bool:
        """Return `True` if this is a error and we must try again"""
        if not isinstance(headers, dict): return True

        headers = {k.lower():v for k, v in headers.items()}
        if "x-ratelimit-remaining" in headers and int(headers['x-ratelimit-remaining']) <= 0:
            super().__api_limit = True
            await asyncio.sleep(int(headers['x-ratelimit-reset']))
            super().__api_limit = False
        return False
               
    async def on_bad_status(self, *args, **kwargs):
        return await self.api_limit(*args, **kwargs)


class substitute:
    """Decorator and context manager to substitute specified exceptions with a certain exception

    With regular function:

    @substitute(Exception, surrogate=NotImplementedError)
    def test():
        raise NotADirectoryError('aaaAaAaa')
    test()
    >>> NotImplementedError: aaaAaAaa

    With coroutine function:

    @substitute(Exception, surrogate=NotImplementedError)
    async def test():
        raise NotADirectoryError
    await test()
    >>> NotImplementedError

    Also this fine works without exceptions:

    @substitute(Exception, surrogate=NotImplementedError)
    async def test():
        return "Hello World!"
    await test()
    >>> Hello World!
    """

    def __init__(self, *exceptions, surrogate):
        self.exceptions = exceptions
        self.surrogate = surrogate

    def __call__(self, func):
        if inspect.iscoroutinefunction(func):
            async def wrapped(*args, **kwargs):
                try: return await func(*args, **kwargs)
                except self.exceptions as e: raise self.surrogate(e)
        else:
            def wrapped(*args, **kwargs):
                try: return func(*args, **kwargs)
                except self.exceptions as e: raise self.surrogate(e)
                    
        return wrapped

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exctype, excinst, exctb):
        raise NotImplementedError

    async def __aenter__(self):
        raise NotImplementedError

    async def __aexit__(self, exctype, excinst, exctb):
        raise NotImplementedError
        

async def execute_cor(*cor):
    """Execute all corutines at the same time and returns result
    
    Example:

    async def foo(a):
        return a
    
    await execute_cor(foo('bar'), foo('buzz'))
    >>> ('bar', 'buzz')
    """

    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(t) for t in cor]
    while True:
        if all([t.done() for t in tasks]):
            return [t.result() for t in tasks]

        await asyncio.sleep(0.05)


# ArgumentParser close program on error, rewrite
class ArgumentParser(argparse.ArgumentParser):    
    def error(self, message):
        raise Exception(message)

    def exit(self, status, message):
        pass


class BaseDiscordBot(discord.Client):
    async def execute_commands(self, message: discord.Message):
        try:
            message_command = message.content.replace(self.command_prefix, '', 1)
            command, args = message_command.split()[0], message_command.split()[1:]
            if message_command in self.commands:
                func = self.commands[message_command]
                args = None

            
            elif command in self.commands:
                func = self.commands[command]
                args = self.parsers[func].parse_args(args)

            else:
                return

            await func(message, args)
            
        except Exception as e:
            traceback.print_exc()
            message = await message.channel.send(e)
            await asyncio.sleep(5)
            await message.delete()

async def sql_table_to_pd(database:str|aiosqlite.Connection, table:str) -> pd.DataFrame:
    async def _exec(db: aiosqlite.Connection):
        db.row_factory = aiosqlite.Row
        async with db.execute(f'SELECT * FROM {table}') as cursor:
            values = await cursor.fetchall()
            async for row in cursor:
                columns = row.keys()
                df = pd.DataFrame(values, columns=columns)
                return df

    if isinstance(database, str):
        async with aiosqlite.connect(database) as db:
            return await _exec(db)

    elif isinstance(database, aiosqlite.Connection):
        return await _exec(database)

    else:
        raise Exception

async def sql_table_to_list(database:str|aiosqlite.Connection, table:str) -> list[dict]:
    """Return table as list of dict, where each list element is table row, dict keys is column names"""
    async def _exec(db: aiosqlite.Connection):
        db.row_factory = aiosqlite.Row
        sql = f"""SELECT * FROM {table}"""
        async with db.execute(sql) as cursor:
            return [{column:value for value, column in zip(row, row.keys())} async for row in cursor]
            
    if isinstance(database, str):
        async with aiosqlite.connect(database) as db:
            return await _exec(db)

    elif isinstance(database, aiosqlite.Connection):
        return await _exec(database)
    
    else:
        raise Exception