import sys; sys.path.append("../")
import traceback

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse


import utils
from utils._typing import api


DEVICE = 'cpu'

app = FastAPI()
loader = utils.development.ModelLoader(DEVICE)
models: dict[str, utils.nn.prematch.PrematchModel] = loader.load_prematch_ensemble_models('2022.08.23 - 21-58', [i for i in range(5)])

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


@app.get("/predict/prematch", response_model=api.Prematch)
async def predict_prematch(team1:int, team2:int, key:str|None=None, match_id:int|None=None) -> dict[str, str] | api.Prematch:
    try:
        if key not in keys: raise Exception("Invalid key")

        wrapper = utils.wrappers.PropertyWrapper()
        data = await wrapper.prematch(team1=team1, team2=team2, match_id=match_id)
        data = utils.nn.tools.batch_to_tensor(data)
        data = utils.nn.tools.batch_to_device(data, DEVICE)

        preds = []
        with torch.no_grad():
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
            OOD_value=std.item(),
            OOD_method='ensemble_std'
        )
 
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=404, content={"message": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)