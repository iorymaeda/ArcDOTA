import sys; sys.path.append("../")


import torch
import uvicorn
from fastapi import FastAPI


import utils


DEVICE = 'cpu'


app = FastAPI()
loader = utils.development.ModelLoader(DEVICE)
models = loader.load_prematch_ensemble_models('2022.08.19 - 14-50', [i for i in range(4)])

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


@app.get("/predict/prematch")
async def predict_prematch(team1:int, team2:int, key:str|None=None, match_id:int|None=None):
    try:
        if key in keys:
            wrapper = utils.wrappers.PropertyWrapper()
            data = await wrapper.prematch(team1=team1, team2=team2, match_id=match_id)
            data = utils.nn.tools.batch_to_tensor(data)
            data = utils.nn.tools.batch_to_device(data, DEVICE)

            preds = []
            with torch.no_grad():
                for m_key in models:
                    model = models[m_key]
                    pred: torch.Tensor = model(data)
                    pred = pred.softmax(dim=1)
                    preds.append(pred[:, 1])

            preds = torch.vstack(preds)
            ensemble_mean_pred = preds.mean(dim=0)
            ensemble_mean_pred = ensemble_mean_pred.item()
            print(ensemble_mean_pred)
            return ensemble_mean_pred
    except Exception as e:
        return e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)