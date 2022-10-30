import time

import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader

from .prematch import PrematchModel
from ..base import ConfigBase
from .tools import to_device


class BaseTrainer(ConfigBase):
    model: nn.Module
    optimizer: torch.optim.Optimizer
    sheduler: torch.optim.lr_scheduler._LRScheduler | None
    device: str
    accum_iter: int
    grad_clip_norm: float
    grad_clip_value: float

    def __init__(self, accum_iter:int=1, grad_clip_norm=0., grad_clip_value=0., **kwargs):
        assert accum_iter >= 1
        assert grad_clip_norm >= 0
        assert grad_clip_value >= 0

        self.epoch = 0
        self.accum_iter = accum_iter
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.model.to(self.device)

        x = to_device(x, self.device)
        pred: torch.Tensor = self.model(x)
        return pred.cpu()

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> list[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.model.to(self.device)

        _pred, _true = [], []
        for batch in dataloader:
            batch = to_device(batch, self.device)
            if isinstance(batch, dict):
                x, y = batch, batch['y']
            elif isinstance(batch, list):
                x, y = batch
            else:
                raise Exception

            pred: torch.Tensor = self.model(x)
            _pred.append(pred.cpu())
            _true.append(y.cpu())

        return torch.cat(_pred), torch.cat(_true)
    
    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError

    def checkpoint(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "configs": {
                "features": self._get_config("features"),
                "match": self._get_config("match"),
                "models": self._get_config("models"),
                "train": self._get_config("train")
            }
        }

    def train_epoch(self, dataloader: DataLoader) -> dict:
        self.model.train()
        self.model.to(self.device)
        self.optimizer.zero_grad()

        step_idx = None
        running_losses = {}
        for batch_idx, batch in enumerate(dataloader):
            batch = to_device(batch, self.device)
            _running_losses = self.train_step(batch)

            if (batch_idx == 0 ):
                running_losses.update(_running_losses)

            for key, value in _running_losses.items():
                running_losses[key] += value

            if ((batch_idx + 1) % self.accum_iter == 0):
                self.update_gradients()
                step_idx = batch_idx

        if step_idx != batch_idx:
            self.update_gradients()

        if self.sheduler: 
            self.sheduler.step()
            
        for key in running_losses:
            running_losses[key] /= batch_idx

        self.epoch += 1
        return running_losses

    def train_step(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    def update_gradients(self):
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.grad_clip_norm, 
                error_if_nonfinite=False
            )
        if self.grad_clip_value > 0:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), 
                clip_value=self.grad_clip_value, 
            )
        self.optimizer.step()
        self.optimizer.zero_grad()


class SupervisedClassification(BaseTrainer):
    metrics_type = None | torchmetrics.Metric | list[torchmetrics.Metric] | dict[str, torchmetrics.Metric]

    def __init__(self, 
        model: PrematchModel, loss_fn: nn.Module, optimizer: nn.Module, 
        sheduler=None, metrics=None, device='cuda', sample_weight=False, r_drop:float=0, 
        c_reg:str|list[str]='none', c_reg_distance:str='cos', c_reg_a:float=0, c_reg_e:int=0, c_reg_detach:bool=False, 
        **kwargs):
        super().__init__(**kwargs)
        assert loss_fn.reduction == 'none'
        assert c_reg_distance in ['cos', 'mse', 'mae']

        self.model: PrematchModel = model.to(device)
        self.device = device
        self.metrics = metrics
        self.metrics = to_device(self.metrics, device)
        self.sample_weight = sample_weight

        self.loss_fn: nn.BCEWithLogitsLoss | nn.CrossEntropyLoss = loss_fn
        self.optimizer: torch.optim.Optimizer = optimizer   
        self.sheduler: torch.optim.lr_scheduler._LRScheduler = sheduler

        self.r_drop = r_drop
        self.c_reg = c_reg
        self.c_reg_a = c_reg_a
        self.c_reg_e = c_reg_e
        self.c_reg_detach = c_reg_detach
        self.c_reg_distance = c_reg_distance
        if self.c_reg != 'none' and not isinstance(self.c_reg, list):
            self.c_reg = [self.c_reg]

    def train_step(self, batch: dict[str, torch.Tensor] | list[torch.Tensor, torch.Tensor]) -> dict:
        if isinstance(batch, dict):
            x, y = batch, batch['y']
        elif isinstance(batch, list):
            x, y = batch
        else:
            raise Exception

        forward_time = time.time()

        outputs: torch.Tensor = self.model(x)
        outputs2: torch.Tensor = None

        # Contrastive regularazation
        contrastive_loss = torch.tensor(0, dtype=outputs.dtype, device=outputs.device)
        if self.c_reg != 'none' and self.c_reg and self.c_reg_a > 0 and self.epoch >= self.c_reg_e:
            embs1 = [self.model.emb_storage[emb_name]  for emb_name in self.c_reg]
            outputs2: torch.Tensor = self.model(x)

            embs2 = []
            for emb_name in self.c_reg:
                _emb: torch.Tensor = self.model.emb_storage[emb_name]
                embs2.append(_emb.detach() if self.c_reg_detach else _emb)

            for idx, (emb1, emb2) in enumerate(zip(embs1, embs2)):
                if self.c_reg_distance == 'cos':
                    cos_sim = 1 - F.cosine_similarity(emb1, emb2, dim=-1)
                    cos_sim = cos_sim.mean()
                    contrastive_loss += cos_sim

                if self.c_reg_distance == 'mse':
                    contrastive_loss += F.mse_loss(emb1, emb2, reduction='mean')

                if self.c_reg_distance == 'mae':
                    contrastive_loss += F.l1_loss(emb1, emb2, reduction='mean')

            contrastive_loss /= (idx + 1)
            contrastive_loss *= self.c_reg_a

        elif self.r_drop > 0:
            outputs2: torch.Tensor = self.model(x)

        # CrossEntropy
        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            y = y.float()
            if len(y.shape) == 1:
                y = y.unsqueeze(1)
            elif len(y.shape) > 2:
                raise Exception(f"bad Y shape: {y.shape}")

            loss: torch.Tensor = self.loss_fn(outputs, y)
            outputs = outputs.sigmoid()

            if outputs2 is not None:
                loss += self.loss_fn(outputs, y)
                loss /= 2
                outputs2 = outputs2.sigmoid()

        elif isinstance(self.loss_fn, nn.CrossEntropyLoss):
            y = y.long()
            if len(y.shape) == 2:
                # |y| : (Batch, Num_clases)
                outputs = outputs.softmax(dim=1)
                loss: torch.Tensor = self.loss_fn(outputs, y)
            elif len(y.shape) == 1:
                # |y| : (Batch)
                loss: torch.Tensor = self.loss_fn(outputs, y)
                outputs = outputs.softmax(dim=1)
            else:
                raise Exception(f"bad Y shape: {y.shape}")

            if outputs2 is not None:
                outputs2 = outputs2.softmax(dim=1)

        else:
            raise Exception("Uncorrect LossFN")

        # Sample weights
        if 'sample_weights' in x and self.sample_weight:
            loss = loss * x['sample_weights']

        # R-drop
        r_drop_loss = torch.tensor(0, dtype=outputs.dtype, device=outputs.device)
        if self.r_drop > 0:
            r_drop_loss = self.r_drop * self.compute_kl_loss(outputs, outputs2)

        loss = loss.mean()
        loss = loss + contrastive_loss + r_drop_loss
        forward_time = time.time() - forward_time

        backward_time = time.time()
        loss.backward()
        backward_time = time.time() - backward_time

        # print(f"{forward_time=}, {backward_time=}")

        with torch.no_grad():
            if isinstance(self.metrics, torchmetrics.Metric):
                self.metrics(outputs, y)

            elif isinstance(self.metrics, list):
                for metric in self.metrics:
                    metric(outputs, y)

            elif isinstance(self.metrics, dict):
                for key in self.metrics:
                    metric = self.metrics[key]
                    metric(outputs, y)

        return {
            'LogLoss': loss.item(),
            'ContrastiveLoss': contrastive_loss.item(),
            'RDrop': r_drop_loss.item(),
        }
    
    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> list[torch.Tensor, torch.Tensor]:
        y_pred, y_true = super().predict(dataloader=dataloader)
        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            y_pred = y_pred.sigmoid()

        elif isinstance(self.loss_fn, nn.CrossEntropyLoss):
            y_pred = y_pred.softmax(dim=1)

        return y_pred, y_true

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader):
        y_pred, y_true = self.predict(dataloader=dataloader)
        metrics = self.compute_metrics(y_pred, y_true)
        return metrics

    @staticmethod
    @torch.no_grad()
    def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred, y_true = y_pred.cpu(), y_true.cpu()
        if len(y_pred.shape) == 1:
            num_classes = 1
        else:
            num_classes = y_pred.shape[1]
        return {
            'Acc': float(torchmetrics.functional.accuracy(y_pred, y_true, num_classes=num_classes)),
            'AUC': float(torchmetrics.functional.auroc(y_pred, y_true, num_classes=num_classes)),
            'LogLoss': metrics.log_loss(y_true, y_pred),
        }
    
    @staticmethod
    def compute_kl_loss(p, q):
        """`p` and `q` is logits after softmax/sigmoid"""
        p_loss = F.kl_div(torch.log(p + 1e-3), torch.log(q + 1e-3), reduction='mean', log_target=True)
        q_loss = F.kl_div(torch.log(q + 1e-3), torch.log(p + 1e-3), reduction='mean', log_target=True)
        loss = (p_loss + q_loss) / 2
        return loss