import torch
import numpy as np
from torch.utils.data import DataLoader


def batch_to_device(batch: dict[str|dict, torch.Tensor|dict], device) \
    -> dict[str|dict, torch.Tensor|dict]:

    def recurssive(batch: dict):
        for k, v in batch.items():
            if isinstance(v, dict):
                batch[k] = recurssive(v)
            elif isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch
    return recurssive(batch)


def batch_to_tensor(batch: dict[str|dict, np.ndarray|dict]) \
    -> dict[str|dict, torch.Tensor|dict]:

    # def recurssive(batch: dict):
    #     for k, v in batch.items():
    #         if isinstance(v, dict):
    #             batch[k] = recurssive(v)
    #         else:
    #             batch[k] = torch.Tensor(v)
    #     return batch
    # return recurssive(batch)

    for batch in (DataLoader([batch], batch_size=1, shuffle=False, num_workers=0)):
        return batch