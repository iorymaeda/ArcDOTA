import asyncio

import torch

import utils


loader = utils.development.ModelLoader('cpu')
models: dict[str, utils.nn.prematch.PrematchModel] = loader.load_prematch_ensemble_models('2022.08.23 - 21-58', [i for i in range(5)])


async def main():
    wrapper = utils.wrappers.PropertyWrapper()
    data = await wrapper.prematch(team1=7119388, team2=15, match_id=6707754788)
    data = utils.nn.tools.batch_to_tensor(data)
    data = utils.nn.tools.batch_to_device(data, 'cpu')
    
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

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
