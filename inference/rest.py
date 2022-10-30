import sys
import time
import pathlib
import traceback
if __name__ == '__main__':
    SCRIPT_DIR = pathlib.Path(__file__).parent
    sys.path.append(str(SCRIPT_DIR.parent))

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse

import utils
import utils.nn
from utils._typing import api


DEVICE = 'cpu'

app = FastAPI()
loader = utils.nn.tools.ModelLoader(DEVICE)
models: dict[str, utils.nn.prematch.PrematchModel] = loader.load_prematch_ensemble_models('2022.10.22 - 21-08', [i for i in range(4)])

keys = {
    "A626C899DE9ABAB14D4EC3312C363": None,
    "898BE8553B474E982FC9B26CDC4C9": None,
    "723EBB6B3A5F9979BA85E36AD66BD": None,
    "31FF1F6996FD8D2F9A8A5ADB997CF": None,
    "3FE13BB3C548F469187151FE6996B": None,
    "13B2C5A66A25A7FD5F8544E7C4A55": None,
}

@app.on_event("startup")
async def startup():
    global wrapper
    # Do this to reopen client session inside async function
    # This is necessary when we use Gunicorn and aiohttp session
    wrapper = utils.wrappers.PropertyWrapper()

@app.get("/")
def read_root():
    return RedirectResponse(url='/docs')

@app.get("/status")
def status():
    pass


@torch.no_grad()
@app.get("/predict/prematch", response_model=api.Prematch | api.Error)
async def predict_prematch(
        key:str=None, team1: int|None = None, team2: int|None = None, 
        league_id: int|None = None, prize_pool: int|None = None, 
        match_id: int|None = None
        ) -> api.Prematch | api.Error:
    """Predict winner"""
    try:
        start_time = time.time()

        if key not in keys: 
            raise Exception("Invalid key")
        if match_id is None and (not team1 or not team2): 
            raise Exception("If you not provide match_id you should provide: team1, team2, [ league_id or prize_pool ]")
        assert (team1 is None and team2 is None) or (team1 is not None and team2 is not None), \
            "Both team1 and team2 should be None or not None"
        assert league_id is not None or prize_pool is not None or match_id is not None, \
            'Please provide at least one argument: league_id, prize_pool, match_id'

        
        data = await wrapper.prematch(
            team1=team1, team2=team2, 
            match_id=match_id, league_id=league_id, 
            prize_pool=prize_pool
            )
        data = utils.nn.tools.batch_to_tensor(data)
        data = utils.nn.tools.to_device(data, DEVICE)

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
        response_time=time.time() - start_time
        return api.Prematch(
            outcome=ensemble_mean_pred,
            response_time=response_time,
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
            ).dict()
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)