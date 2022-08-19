import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import ConfigBase
from .tools import batch_to_device


class BaseTrainer(ConfigBase):
    optimizer: torch.optim.Optimizer
    sheduler: torch.optim.lr_scheduler._LRScheduler | None
    model: nn.Module
    device: str

    def predict(self, dataset: DataLoader) -> list[torch.Tensor]:
        self.model.eval()
        self.model.to(self.device)

        _pred, _true = [], []
        with torch.inference_mode():
            for batch in dataset:
                batch = batch_to_device(batch, self.device)
                X, Y = batch, batch['y']

                pred: torch.Tensor = self.model(X)
                _pred+= [pred.cpu()]
                _true+= [Y.cpu()]
        return torch.cat(_pred), torch.cat(_true)


    def checkpoint(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "configs": {
                "features": self._get_config("features"),
                "match": self._get_config("match"),
                "models": self._get_config("models"),
                "train": self._get_config("train")}}

    def train_epoch(self, dataset: DataLoader) -> float:
        self.model.train()
        self.model.to(self.device)

        running_loss = 0
        for idx, batch in enumerate(dataset):
            running_loss+= self.train_step(batch)

        if self.sheduler: self.sheduler.step()
        return running_loss/idx

    def train_step(self, *args, **kwargs):
        raise NotImplementedError


class PremtachTrainer(BaseTrainer):
    loss_fns = nn.BCEWithLogitsLoss | nn.CrossEntropyLoss
    metrics = None | torchmetrics.Metric | list[torchmetrics.Metric] | dict[str, torchmetrics.Metric]

    def __init__(self, 
        model, loss_fn, optimizer, sheduler=None,
        metric=None, device='cuda', **kwargs):

        
        self.model: nn.Module = model.to(device)
        self.device = device
        self.metric: self.metrics = metric
        
        self.loss_fn: self.loss_fns = loss_fn
        self.optimizer: torch.optim.Optimizer = optimizer   
        self.sheduler: torch.optim.lr_scheduler._LRScheduler = sheduler
        
        # --------------------------------------------- #
        # Move metrics to device
        metric: torchmetrics.Metric
        if isinstance(self.metric, torchmetrics.Metric):
            self.metric.to(device)

        elif isinstance(self.metric, list):
            for metric in self.metric:
                metric.to(device)

        elif isinstance(self.metric, dict):
            for key in self.metric:
                metric = self.metric[key]
                metric.to(device)
            

    def train_step(self, batch: dict[torch.Tensor|dict, torch.Tensor|dict]):
        batch = batch_to_device(batch, self.device)
        X, Y = batch, batch['y']
        
        outputs: torch.Tensor = self.model(X)

        if len(Y.shape) > 1:
            outputs = outputs.softmax(dim=1)

        loss: torch.Tensor = self.loss_fn(outputs, Y)    

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        with torch.no_grad():
            # ------------------------------------------------ #
            if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
                outputs, Y = outputs.sigmoid(), Y.int()

            elif isinstance(self.loss_fn, nn.CrossEntropyLoss):
                outputs, Y = outputs.softmax(dim=1), Y.int()

            # ------------------------------------------------ #
            # if len(outputs.shape) > 1 and outputs.shape[1] == 2:
            #     outputs = outputs[:, 1]

            # if len(outputs.shape) > 1 and outputs.shape[1] == 1:
            #     outputs = outputs.squeeze(1)

            # ------------------------------------------------ #
            if isinstance(self.metric, torchmetrics.Metric):
                self.metric(outputs, Y)

            elif isinstance(self.metric, list):
                for metric in self.metric:
                    metric(outputs, Y)

            elif isinstance(self.metric, dict):
                for key in self.metric:
                    metric = self.metric[key]
                    metric(outputs, Y)

        return loss.item()


