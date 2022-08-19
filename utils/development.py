"""Stranger thins to do develompent stuff"""
import time
import asyncio
import datetime
import traceback
import pathlib

import torch
import aiohttp

from .exceptions import opendota
from .base import PathBase, ConfigBase
from .nn.prematch import PrematchModel


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

        if you need to trigger async function `bar` at the end:
        with suppress(FileNotFoundError, triger=bar()):
             os.remove(somefile)

    """
    def __init__(self, *exceptions, print_exc=False, trigger=None):
        self.exceptions = exceptions
        self.print_exc = print_exc
        self.trigger = trigger
        

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        if exctype is not None:
            self.trigger() if self.trigger else None
            if self.print_exc: traceback.print_exc() 

        # Return True for working well or False to raise exception
        return exctype is not None and issubclass(exctype, self.exceptions)

    async def __aenter__(self):
        pass

    async def __aexit__(self, exctype, excinst, exctb):
        if exctype is not None:
            await self.trigger if self.trigger else None
            if self.print_exc: traceback.print_exc() 
            
        # Return True for working well or False to raise exception
        return exctype is not None and issubclass(exctype, self.exceptions)


class SessionHelper:
    def __init__(self):
        self.timeout = aiohttp.ClientTimeout(total=61)
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        self.verbose = True

    @property
    def __current_sec(self) -> int:
        date = datetime.datetime.now()
        return date.second


    async def opendota_api_limit(self, sec:int , headers: dict) -> bool | None:
        """Return `True` if this is a opendota time limit and we must try again"""
        if not isinstance(headers, dict): return
        if ("x-rate-limit-remaining-month" in headers and 
            "x-rate-limit-remaining-minute" in headers):
            if int(headers['x-rate-limit-remaining-month']) <= 0:
                raise opendota.OpendotaMonthLimit

            if int(headers['x-rate-limit-remaining-minute']) <= 0:   
                if self.verbose: print(f"Opendota rate limit, sleep: {60-sec + 1}")
                # If a new minute has been begun after request - limit has skiped and
                # `__current_sec` will be less than `sec`
                if self.__current_sec > sec: 
                    time.sleep(60-sec + 1)
                    # await asyncio.sleep(60-sec + 1)

                return True
                

    async def get(self, URL: str, counter=0, json=None, status=666): 
        if counter > 3: return json, status

        try:
            sec = self.__current_sec
            response = await self.session.get(URL)
            status = response.status
            if status//100 in [2, 3]:
                json = await response.json()
                return json, status
            else:
                if not await self.opendota_api_limit(sec, response.headers):
                    counter+=1

                json, status = await self.get(URL=URL, counter=counter, json=json, status=status)
                return json, status

        except (aiohttp.ServerTimeoutError, asyncio.exceptions.TimeoutError):
            json, status = await self.get(URL=URL, counter=counter+1, json=json, status=status)
            return json, status


    async def post(self, URL: str, counter=0, json=None, status=666): 
        if counter > 3: return json, status

        try:
            sec = self.__current_sec
            response = await self.session.post(URL)
            status = response.status
            if status//100 in [2, 3]:
                json = await response.json()
                return json, status
            else:
                if not await self.opendota_api_limit(sec, response.headers):
                    counter+=1

                json, status = await self.post(URL=URL, counter=counter+1, json=json, status=status)
                return json, status

        except (aiohttp.ServerTimeoutError, asyncio.exceptions.TimeoutError):
            json, status = await self.post(URL=URL, counter=counter+1, json=json, status=status)
            return json, status
            

    async def close(self):
        async with suppress(Exception):
            await self.session.close()


    async def reset(self):
        await self.close()
        self.session = aiohttp.ClientSession(timeout=self.timeout)


class ModelLoader(PathBase):
    def __init__(self, device:str='cpu'):
        self.device = device

    def __get_weights_folder(self):
        # Project folder
        # ├── train
        # │   ├── output
        # │   │   ├──models_w
        # │   │   │  ├── prematch
        # │   │   │  └──...
        # │   │   └──...
        # │   └──...
        # ├── utils
        # │   ├── development.py
        # │   └──...
        path = self._get_curernt_path()
        # -> development.py
        path = path.parent.absolute() 
        # -> Project folder
        path = path.joinpath("train/output/models_w/") 
        # -> prematch
        return path

    def __get_prematch_folder(self):
        path = self.__get_weights_folder()
        path = path.joinpath("prematch/") 
        return path


    def load_prematch_ensemble_models(self, name: str, ensemble_nums: list, device:str=None):
        if device is None: device = self.device

        path = self.__get_prematch_folder()
        models = {}
        for num in ensemble_nums:
            checkpoint = torch.load(path / f'Ensemble {num} {name}.torch')
            for config in checkpoint['configs']:
                ConfigBase._configs[config] = checkpoint['configs'][config]
                
            model = PrematchModel(**checkpoint['kwargs'])
            model.to(device)
            model.load_state_dict(checkpoint['model'])
            models[num] = model
        return models


