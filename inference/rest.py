import sys; sys.path.append("../")
import traceback
from seleniumwire import webdriver  # Import from seleniumwire
import seleniumwire.undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse


import utils
from utils._typing import api


DEVICE = 'cpu'

app = FastAPI()
loader = utils.nn.tools.ModelLoader(DEVICE)
models: dict[str, utils.nn.prematch.PrematchModel] = loader.load_prematch_ensemble_models('2022.09.14 - 21-46', [i for i in range(5)])

keys = {
    "A626C899DE9ABAB14D4EC3312C363": None,
    "898BE8553B474E982FC9B26CDC4C9": None,
    "723EBB6B3A5F9979BA85E36AD66BD": None,
    "31FF1F6996FD8D2F9A8A5ADB997CF": None,
    "3FE13BB3C548F469187151FE6996B": None,
    "13B2C5A66A25A7FD5F8544E7C4A55": None,
}

@app.get("/")
def read_root():
    return "This is ArcDota api, see /docs for more info"

@torch.no_grad()
@app.get("/predict/prematch", response_model=api.Prematch | api.Error)
async def predict_prematch(team1:int, team2:int, key:str, match_id:int|None=None) \
    -> api.Prematch | api.Error:
    try:
        if key not in keys: raise Exception("Invalid key")

        wrapper = utils.wrappers.PropertyWrapper()
        data = await wrapper.prematch(team1=team1, team2=team2, match_id=match_id)
        data = utils.nn.tools.batch_to_tensor(data)
        data = utils.nn.tools.batch_to_device(data, DEVICE)

        preds = []
        for m_key in models:
            model = models[m_key]
            if model.regression: raise NotImplementedError

            pred: torch.Tensor = model.predict(data)
            preds.append(pred)

        preds = torch.vstack(preds)
        std = preds.std(dim=0)

        ensemble_mean_pred = preds.mean(dim=0)
        ensemble_mean_pred = ensemble_mean_pred.item()
        return api.Prematch(
            outcome=ensemble_mean_pred,
            OOD_method_value=[
                api.OOD(method='ESTD', score=std.item()),
                api.OOD(method='DIME', score=0),
            ]
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=404,
            content=api.Error(
                status_code=404,
                message=str(e),
                error=str(type(e).__name__)
            )
        )


@app.get("/hawk/get-matches")
async def hawk_get_matches(date:str):
    try:
        chrome_options = uc.ChromeOptions()

        # Create a new instance of the Chrome driver (or Firefox)
        driver = uc.Chrome(
            options=chrome_options,
            seleniumwire_options={}
        )

        # Create a request interceptor
        def interceptor(request):
            request.headers['accept'] = 'text/html, application/xhtml+xml'
            request.headers['accept-encoding'] = 'gzip, deflate, br'
            request.headers['accept-language'] = 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7'
            request.headers['content-type'] = 'application/json'
            request.headers['cookie'] = 'cf_clearance=eIcEuSEVRZqi.ou6gG0Pee7Ieq0u4eIDKbI3x421uG0-1666208016-0-150; _ym_uid=1666208019767534247; _ym_d=1666208019; _ym_isad=1; _ym_visorc=w; _gid=GA1.2.1773172140.1666208019; XSRF-TOKEN=eyJpdiI6Ing2TnVNVWJ5dmVhUTkyYkVSZkt3Q1E9PSIsInZhbHVlIjoiM3lpY2ZYbEJwdHh3VDdDemM1Z1VGVjlncUUyWjNKUHFVWlViZTIvWXlUZk90S0Q4QUxjakRUM3phdUZXMWFKMlpPV3pIL2pRaFQ3a1A1VGV5RUxNN0lPV29MN2ZBclhyNjVKekVWUkcvSER5MzhEVlFQRUVidmVqcHdsOFJTbUEiLCJtYWMiOiI4MWQ3M2VkZTlhZmRlMzdlZDI5MWI2OWU2NzhlNjIzNGNmOWUwY2YzNjM5YzkzM2QxNmUxNzk5Njg4N2RjYWM5IiwidGFnIjoiIn0%3D; hawk_session=eyJpdiI6IlFlOFdESE00OXZiazBJQUVaYi9hUEE9PSIsInZhbHVlIjoiL3RSeUxycW54djR2WlBFYTBkY09rdW42NE4xQ3hNaU9aVHJNRll6dngxYmpPWDdaZC9uSWl0UFA2SW5BbHk2MWhQR01Sb0RNL3FJZEZPYktZeXpZZUVOaDRWMzdsUk84ZTlMbVBqNHJnaUNlZVBsOVc2MWh4OWNxV3NWV1VDaGgiLCJtYWMiOiIyODdmZGZmOTgyMjRmNWEzNmFiYTEwMDBiNTczZGRiYmIxM2MyNDc5MDBhMTEwNTVkMmZiYjVlYTc3MWM3MWM2IiwidGFnIjoiIn0%3D; _ga_2751SH5P8F=GS1.1.1666208019.1.1.1666208969.0.0.0; _ga=GA1.1.1354790473.1666208019; io=jKsvY21rBl6-R9YBOoUa'
            request.headers['referer'] = 'https://hawk.live/'
            request.headers['sec-ch-ua'] = '"Chromium";v="106", "Google Chrome";v="106", "Not;A=Brand";v="99"'
            request.headers['sec-ch-ua-mobile'] = '?0'
            request.headers['sec-ch-ua-platform'] = '"Windows"'
            request.headers['sec-fetch-dest'] = 'empty'
            request.headers['sec-fetch-mode'] = 'cors'
            request.headers['sec-fetch-site'] = 'same-origin'
            request.headers['user-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'
            request.headers['x-inertia'] = 'true'
            request.headers['x-inertia-version'] = 'b68330e61c61ba0f7b85100b16f22e48'
            request.headers['x-requested-with'] = 'XMLHttpRequest'
            #request.headers['x-xsrf-token'] = 'eyJpdiI6Ik12S0wrWHJjQ3lIeWpXc2dNQ2QyNUE9PSIsInZhbHVlIjoiWmpPc0hMWVZ2aU8yWGc5ZC83bHN2c041cHB6R2pnck1rbURjekNydTlTcjgxS0lEalFQOEM4c01OOHFvRkdUNDZ6T1VvU1YxQkxmV2ZiTzZpb3hDUE1wc0E0SmthaE1JcGRJS245NEZhK0tBeUhMaVE0WmkrQit6Qkx1ZlpxNWciLCJtYWMiOiI1MmYyMTZlMmQyOWRhYzE0ZTJhYWNmMjU2OWM3ZTVlYTg1ZDUwNjUyMDljNzBkMDI5ZGIwNmE5ODQ2MjdjOWQ2IiwidGFnIjoiIn0='


        # Set the interceptor on the driver
        driver.request_interceptor = interceptor

        driver.get('view-source:https://hawk.live/matches/recent/2022-10-19')
        content = driver.page_source
        driver.close()

        test = content[338:]
        test = test[0:len(test)-78]

        return test

    except Exception as e:
        return JSONResponse(
            status_code=404,
            message=str(e),
            error=str(type(e).__name__)
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
