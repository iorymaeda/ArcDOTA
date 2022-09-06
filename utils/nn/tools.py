import torch
import numpy as np
from torch.utils.data import DataLoader

from ..base import PathBase, ConfigBase
from .prematch import PrematchModel


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


class ModelLoader(PathBase):
    #//TODO: move this to nn
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
        path = path.parent.resolve() 
        path = path / "train/output/models_w/"
        return path

    def __get_inference_weights_folder(self):
        # Project folder
        # ├── inference
        # │   ├── models_w
        # │   │   ├── prematch
        # │   │   └──...
        # │   └──...
        # ├── utils
        # │   ├── development.py
        # │   └──...
        path = self._get_curernt_path()
        path = path.parent.resolve()
        path = path / "inference/models_w/"
        return path

    def __get_prematch_folder(self):
        path = self.__get_inference_weights_folder()
        path = path / "prematch/"
        return path


    def load_prematch_ensemble_models(self, name: str, ensemble_nums: list, device:str=None):
        if device is None: device = self.device

        path = self.__get_prematch_folder()
        models = {}
        for num in ensemble_nums:
            checkpoint = torch.load(path / f'Ensemble {num} {name}.torch', map_location=torch.device('cpu'))
            for config in checkpoint['configs']:
                ConfigBase._configs[config] = checkpoint['configs'][config]
                
            model = PrematchModel(**checkpoint['kwargs']).eval()
            model.to(device)
            model.load_state_dict(checkpoint['model'])
            models[num] = model
        return models