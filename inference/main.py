import sys; sys.path.append("../")


import torch
import uvicorn
from fastapi import FastAPI


import utils


DEVICE = 'cuda'


app = FastAPI()
loader = utils.development.ModelLoader(DEVICE)
models = loader.load_prematch_ensemble_models('2022.08.18 - 22-07', [i for i in range(4)])


@app.get("/")
def read_root():
    return "This is ArcDota api, see /docs for more info"


@app.get("/predict/prematch")
async def predict_prematch(team1:int, team2:int):
    wrapper = utils.wrappers.PropertyWrapper()
    data = await wrapper.prematch(team1=team1, team2=team2)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)