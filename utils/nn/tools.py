import glob

import torch
import numpy as np
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader

from .prematch import PrematchModel
from ..base import PathBase, ConfigBase


def to_device(batch: list|torch.Tensor|dict|torchmetrics.Metric, device) \
    -> list|torch.Tensor|dict:
    """Move any type of batch|nn.Module to device"""
    def recurssive_dict(batch: dict):
        for k, v in batch.items():
            if isinstance(v, dict):
                batch[k] = recurssive_dict(v)
            elif isinstance(v, (list, tuple)):
                batch[k] = recurssive_list(v)
            elif isinstance(v, (torch.Tensor, torchmetrics.Metric)):
                batch[k] = v.to(device)
        return batch
    
    def recurssive_list(batch: list):
        batch = list(batch)
        for idx, item in enumerate(batch):
            if isinstance(item, (list, tuple)):
                batch[idx] = recurssive_list(item)
            elif isinstance(item, dict):
                batch[idx] = recurssive_dict(item)
            elif isinstance(item, (torch.Tensor, torchmetrics.Metric)):
                batch[idx] = item.to(device)
        return batch

    if batch is None:
        return None
    elif isinstance(batch, dict):
        return recurssive_dict(batch)
    elif isinstance(batch, (list, tuple)):
        return recurssive_list(batch)
    elif isinstance(batch, (torch.Tensor, torchmetrics.Metric, torch.nn.Module)):
        return batch.to(device)
    else:
        raise Exception("Unexcepted batch type:", type(batch))

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
    def __init__(self, device:str='cpu'):
        self.device = device

    def __get_weights_folder(self):
        """
        Project folder
        ├── train
        │   ├── output
        │   │   ├──models_w
        │   │   │  ├── prematch
        │   │   │  └──...
        │   │   └──...
        │   └──...
        ├── utils
        │   ├── development.py
        │   └──...
        """
        path = self._get_relative_path()
        path = path.parent.resolve() 
        path = path / "train/output/models_w/"
        return path

    def __get_inference_weights_folder(self):
        """
        Project folder
        ├── inference
        │   ├── files
        │   │   ├── models_w
        │   │   │   ├──prematch
        │   │   │   └──...
        │   │   └──...
        │   └──...
        ├── utils
        │   ├── development.py
        │   └──...
        """
        path = self._get_relative_path()
        path = path.parent.resolve()
        path = path / "inference/files/models_w/"
        return path

    def __get_prematch_folder(self):
        path = self.__get_inference_weights_folder()
        path = path / "prematch/"
        return path


    def load_prematch_ensemble_models(self, device:str=None) -> dict[str, nn.Module]:
        if device is None: device = self.device

        path = self.__get_prematch_folder()
        files = glob.glob(f'{path}/*.torch')

        models = {}
        for idx, file in enumerate(files):
            checkpoint = torch.load(file, map_location=torch.device('cpu'))
            ConfigBase._configs =  checkpoint['configs']
            model = PrematchModel(**checkpoint['kwargs']).eval()
            model.to(device)

            try:
                model.load_state_dict(checkpoint['model'])
            except Exception as e:
                if 'missing' in str(e).lower():
                    raise Exception(e)
                else:
                    print("Unexcepted w., load without strict")
                    model.load_state_dict(checkpoint['model'], strict=False)

            models[checkpoint['model_tag']] = model
        return models

# def no_dropout(x): return x
no_dropout = nn.Identity()
no_dropout.p = 0

# def no_layer_norm(x): return x
no_layer_norm = nn.Identity()


def get_indicator(length_tensor, max_length=None):
    """
    :param length_tensor: 
    :param max_length: 
    :returns: a tensor where positions within ranges are set to 1
    """
    lengths_size = length_tensor.size()

    flat_lengths = length_tensor.view(-1, 1)

    if not max_length:
        max_length = length_tensor.max()
    unit_range = torch.arange(max_length)
    # flat_range = torch.stack([unit_range] * flat_lengths.size()[0],
    #                          dim=0)
    # flat_range = unit_range.repeat(flat_lengths.size()[0], 1)
    flat_range = unit_range.expand(flat_lengths.size()[0:1] + unit_range.size())
    flat_indicator = flat_range < flat_lengths

    return flat_indicator.view(lengths_size + (-1, 1))


def create_lstm_cell_init_state(hidden_size, init_state_learned=True):
    """
    :param hidden_size: 
    :param init_state_learned: 
    :returns: init_state is a input of lstm cells. _init_state is saved as a parameter of model (such as self._init_state)
    """
    init_hidden = nn.Parameter(torch.zeros(1, hidden_size), init_state_learned)
    init_cell = nn.Parameter(torch.zeros(1, hidden_size), init_state_learned)

    init_state = (init_hidden, init_cell)
    _init_state = nn.ParameterList(init_state)

    return init_state, _init_state


def repeat_lstm_cell_state(state, batch_size):
    for s in state:
        size = s.size()
        assert len(size) == 2
    # s is either hidden or cell
    return tuple(
        # s.repeat(batch_size, 1)
        s.squeeze(0).expand((batch_size,) + s.size()[1:])
        for s in state)


def create_lstm_init_state(num_layers, num_directions, hidden_size, init_state_learned=True):
    """
    :param hidden_size: 
    :param init_state_learned: 
    :returns: init_state is a input of lstm cells. _init_state is saved as a parameter of model (such as self._init_state)
    """
    init_hidden = nn.Parameter(torch.zeros(
        num_layers * num_directions, 1, hidden_size), init_state_learned)
    init_cell = nn.Parameter(torch.zeros(num_layers * num_directions,
                                         1, hidden_size), init_state_learned)

    init_state = (init_hidden, init_cell)
    _init_state = nn.ParameterList(init_state)

    return init_state, _init_state


def repeat_lstm_state(state, batch_size):
    # s is either hidden or cell
    return tuple(
        s.repeat(1, batch_size, 1)
        for s in state)


def is_cuda_enabled(model):
    return next(model.parameters()).is_cuda


def get_module_device(model):
    return next(model.parameters()).device