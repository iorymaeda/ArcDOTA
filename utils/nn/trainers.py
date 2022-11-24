import time

import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader

from .prematch import PrematchModel
from ..base import ConfigBase
from .tools import to_device, cat_batches


class BaseTrainer:
    model: nn.Module | list[nn.Module]
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
    def forward(self, x: torch.Tensor|dict[str, torch.Tensor]|list[torch.Tensor]) -> torch.Tensor:
        """forward `x` over model/models

        returns `torch.Tensor` of shape: (num_of_models, batch_size, output_dimension)
        """
        if isinstance(self.model, list):
            models = [model for model in self.model]
        else:
            models = [self.model]

        preds = []
        for model in models:
            x = to_device(x, self.device)
            pred: torch.Tensor = model(x)
            preds.append(pred)

        pred = torch.stack(preds)
        # |pred| : (num_of_models, batch_size, output_dimension)
        return pred

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> list[torch.Tensor, torch.Tensor]:
        if isinstance(self.model, list):
            self.model = [model.eval().to(self.device) for model in self.model]
        else:
            self.model.eval().to(self.device)

        _pred, _true = [], []
        for batch in dataloader:
            batch = to_device(batch, self.device)
            if isinstance(batch, dict):
                x, y = batch, batch['y']
            elif isinstance(batch, list):
                x, y = batch
            else:
                raise Exception

            pred = self.forward(x).mean(0)
            _pred.append(pred)
            _true.append(y)
        
        pred = torch.cat(_pred).cpu()
        true = torch.cat(_true).cpu()
        return pred, true
    
    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError

    def checkpoint(self) -> dict | list[dict]:
        if isinstance(self.model, list):
            checkpoints = [{
                "model": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "configs": model.configs,
                } for model in self.model]
            return checkpoints
        else:
            return {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "configs": self.model.configs,
            }

    def train_epoch(self, dataloader: DataLoader) -> dict:
        if isinstance(self.model, list):
            self.model = [model.train().to(self.device) for model in self.model]
        else:
            self.model.train().to(self.device)

        self.optimizer.zero_grad()

        step_idx = None
        running_losses = {}
        for batch_idx, batch in enumerate(dataloader):
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
        batch = to_device(batch, self.device)
        raise NotImplementedError

    def update_gradients(self):
        if isinstance(self.model, list):
            models = [model for model in self.model]
        else:
            models = [self.model]

        for model in models:
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=self.grad_clip_norm, 
                    error_if_nonfinite=False
                )
            if self.grad_clip_value > 0:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), 
                    clip_value=self.grad_clip_value, 
                )
        self.optimizer.step()
        self.optimizer.zero_grad()


class SupervisedClassificationTrainer(BaseTrainer):
    metrics_type = None | torchmetrics.Metric | list[torchmetrics.Metric] | dict[str, torchmetrics.Metric]

    def __init__(self, 
        model: PrematchModel | list[PrematchModel], loss_fn: nn.Module, optimizer: nn.Module, 
        sheduler=None, metrics=None, device='cuda', sample_weight=False, r_drop:float=0, 
        c_reg:str|list[str]='none', c_reg_distance:str='cos', c_reg_a:float=0, c_reg_e:int=0, c_reg_detach:bool=False,
        boosting_c_reg_a: float=0, boosting_c_reg_e: int=0, boosting_c_reg_distance:str='mse', 
        backward_second_output: bool = False,
        **kwargs):
        super().__init__(**kwargs)
        assert loss_fn.reduction == 'none'
        assert c_reg_distance in ['cos', 'mse', 'mae']
        assert boosting_c_reg_distance in ['cos', 'mse', 'mae']

        self.device = device
        self.metrics = metrics
        self.metrics = to_device(self.metrics, device)
        self.sample_weight = sample_weight

        self.loss_fn: nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | nn.L1Loss | nn.MSELoss = loss_fn
        self.optimizer: torch.optim.Optimizer = optimizer   
        self.sheduler: torch.optim.lr_scheduler._LRScheduler = sheduler

        self.r_drop = r_drop
        self.c_reg = c_reg
        self.c_reg_a = c_reg_a
        self.c_reg_e = c_reg_e
        self.c_reg_detach = c_reg_detach
        self.c_reg_distance = c_reg_distance
        self.boosting_c_reg_a = boosting_c_reg_a
        self.boosting_c_reg_e = boosting_c_reg_e
        self.boosting_c_reg_distance = boosting_c_reg_distance
        self.backward_second_output = backward_second_output
        if self.c_reg != 'none' and not isinstance(self.c_reg, list):
            self.c_reg = [self.c_reg]

        self.сontrastive_condition = self.c_reg != 'none' and self.c_reg and self.c_reg_a > 0
        self.boosting_condition =  self.c_reg != 'none' and boosting_c_reg_a > 0
        if (not self.сontrastive_condition and self.r_drop == 0 and not self.boosting_condition) and backward_second_output:
            print(r'Warning: We can not do backward pass on second output, cause we have only one output from the model (use сontrastive regulation or r_drop). So we will forward batch from model/models twice')

        if isinstance(model, list):
            if self.boosting_condition and len(model) == 1:
                raise Exception(r'You tried to use cross ensemble сontrastive reguralizatoin which means you minimize `boosting_c_reg_distance` beetwen this models layers, but there only one model')

            self.model = [m.to(device) for m in model]
            self.mode = 'ensemble'

        else:    
            if self.boosting_condition:
                raise Exception(r'You tried to use cross ensemble сontrastive reguralizatoin which means you minimize `boosting_c_reg_distance` beetwen this models layers, but there only one model')

            self.model: PrematchModel = model.to(device)
            self.mode = 'normal'

    def train_step(self, batch: dict[str, torch.Tensor] | list[torch.Tensor, torch.Tensor]) -> dict:
        batch = to_device(batch, self.device)
        сontrastive_condition = self.сontrastive_condition and self.epoch >= self.c_reg_e
        boosting_condition = self.boosting_condition and self.epoch >= self.boosting_c_reg_e

        if isinstance(batch, dict):
            x, y = batch, batch['y']
        elif isinstance(batch, list):
            x, y = batch
        else:
            raise Exception

        forward_time = time.time()

        if self.mode == 'normal':
            models = [self.model]
        elif self.mode == 'ensemble':
            models = self.model
            
        # Forward batch twice
        if сontrastive_condition or boosting_condition or self.r_drop > 0 or self.backward_second_output:
            x = cat_batches(x, x) #inplace operation
            stacked_outputs: list[torch.Tensor] = [model(x) for model in models]
            stacked_outputs: torch.Tensor = torch.stack(stacked_outputs).mean(dim=0)

            n = len(stacked_outputs)//2
            outputs, outputs2 = stacked_outputs[:n], stacked_outputs[n:]

        else:
            outputs: list[torch.Tensor] = [model(x) for model in models]
            outputs: torch.Tensor = torch.stack(outputs).mean(dim=0)
            n = len(outputs)

        # 0 - list with activation rom different layers in first model
        # ...
        # n - list with activation rom different layers in n model
        activations_over_models = []
        # Contrastive regularazation
        cl_loss = torch.tensor(0, dtype=outputs.dtype, device=outputs.device)
        if сontrastive_condition:
            for model in models:
                model: PrematchModel
                embs1 = [model.emb_storage[emb_name][:n] for emb_name in self.c_reg]
                embs2 = [model.emb_storage[emb_name][n:] for emb_name in self.c_reg]
                activations_over_models.append(embs1)

                if self.c_reg_detach:
                    embs2 = [e.detach() for e in embs2]
                
                for idx, (emb1, emb2) in enumerate(zip(embs1, embs2)):
                    if self.c_reg_distance == 'cos':
                        cos_sim = 1 - F.cosine_similarity(emb1, emb2, dim=-1)
                        cos_sim = cos_sim.mean()
                        cl_loss += cos_sim

                    if self.c_reg_distance == 'mse':
                        cl_loss += F.mse_loss(emb1, emb2, reduction='mean')

                    if self.c_reg_distance == 'mae':
                        cl_loss += F.l1_loss(emb1, emb2, reduction='mean')

                cl_loss /= (idx + 1)
                cl_loss /= len(models)
        
        cb_loss = torch.tensor(0, dtype=outputs.dtype, device=outputs.device)
        if boosting_condition:
            # Cross Boosting contrastive regularazation 
            if not activations_over_models:
                for model in models:
                    embs = [model.emb_storage[emb_name][:n] for emb_name in self.c_reg]
                    activations_over_models.append(embs)
                    
            for model_num1, activations1 in enumerate(activations_over_models):
                for model_num2, activations2 in enumerate(activations_over_models):

                    if model_num1 != model_num2:
                        for act1, act2 in zip(activations1, activations2):
                            if self.boosting_c_reg_distance == 'cos':
                                cos_sim = 1 - F.cosine_similarity(act1, act2, dim=-1)
                                cos_sim = cos_sim.mean()
                                cb_loss += cos_sim

                            if self.boosting_c_reg_distance == 'mse':
                                cb_loss += F.mse_loss(act1, act2, reduction='mean')

                            if self.boosting_c_reg_distance == 'mae':
                                cb_loss += F.l1_loss(act1, act2, reduction='mean')

            cb_loss /= (model_num1+1)
            cb_loss /= (model_num2+1)
            cb_loss /= len(activations1)

        # CrossEntropy
        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            y = y.float()
            if len(y.shape) == 1:
                y = y.unsqueeze(1)
            elif len(y.shape) > 2:
                raise Exception(f"bad Y shape: {y.shape}")

            loss: torch.Tensor = self.loss_fn(outputs, y)
            outputs = outputs.sigmoid()
            if self.backward_second_output:
                loss = (loss + self.loss_fn(outputs2, y)) / 2
                outputs2 = outputs2.sigmoid()

        elif isinstance(self.loss_fn, nn.CrossEntropyLoss):
            y = y.long()
            if len(y.shape) == 2:
                # |y| : (Batch, Num_clases)
                outputs = outputs.softmax(dim=1)
                loss: torch.Tensor = self.loss_fn(outputs, y)
                if self.backward_second_output:
                    outputs2 = outputs2.softmax(dim=1)
                    loss = (loss + self.loss_fn(outputs2, y)) / 2
                    
            elif len(y.shape) == 1:
                # |y| : (Batch)
                loss: torch.Tensor = self.loss_fn(outputs, y)
                outputs = outputs.softmax(dim=1)
                if self.backward_second_output:
                    loss = (loss + self.loss_fn(outputs2, y)) / 2
                    outputs2 = outputs2.softmax(dim=1)

            else:
                raise Exception(f"bad Y shape: {y.shape}")

        else:
            # ---------------------------------------------- #
            y = y.float()
            if (y.ndim == 2 and y.size(1) == 1) or (y.ndim == 1 and outputs.size(1) == 1):
                outputs = outputs.sigmoid()
                if self.backward_second_output:
                    outputs2 = outputs2.sigmoid()
            elif (y.ndim == 2 and y.size(1) > 1) or (y.ndim == 1 and outputs.size(1) > 1):
                outputs = outputs.softmax(dim=-1)
                if self.backward_second_output:
                    outputs2 = outputs2.softmax(dim=-1)

            # ---------------------------------------------- #
            # Compare dimensions 
            if outputs.ndim == 2 and y.ndim == 2:
                if outputs.size(1) == 2 and y.size(1) == 1:
                    outputs = outputs[:, 1:]
                    outputs2 = outputs2[:, 1:] if self.backward_second_output else None

            elif outputs.ndim == 2 and y.ndim == 1:
                outputs = outputs[:, 1]
                outputs2 = outputs2[:, 1] if self.backward_second_output else None

            loss: torch.Tensor = self.loss_fn(outputs, y)
            if self.backward_second_output:
                loss = (loss + self.loss_fn(outputs2, y)) / 2

        # Sample weights
        if 'sample_weights' in x and self.sample_weight:
            loss = loss * x['sample_weights'][:n]

        # R-drop
        r_drop_loss = torch.tensor(0, dtype=outputs.dtype, device=outputs.device)
        if self.r_drop > 0:
            r_drop_loss += self.compute_kl_loss(outputs, outputs2)

        loss = loss.mean()
        loss = loss + cl_loss*self.c_reg_a + cb_loss*self.boosting_c_reg_a + r_drop_loss*self.r_drop
        forward_time = time.time() - forward_time

        backward_time = time.time()
        loss.backward()
        backward_time = time.time() - backward_time
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
            'ContrastiveLoss': cl_loss.item(),
            'CBContrastiveLoss': cb_loss.item(),
            'RDrop': r_drop_loss.item(),
            'forward_time': forward_time,
            'backward_time': backward_time,
        }

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> list[torch.Tensor, torch.Tensor]:
        y_pred, y_true = super().predict(dataloader=dataloader)
        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            y_pred = y_pred.sigmoid()
            
        elif isinstance(self.loss_fn, nn.CrossEntropyLoss):
            y_pred = y_pred.softmax(dim=1)
        
        else:
            if (y_true.ndim == 2 and y_true.size(1) == 1) or (y_true.ndim == 1 and y_pred.size(1) == 1):
                y_pred = y_pred.sigmoid()
            elif (y_true.ndim == 2 and y_true.size(1) > 1) or (y_true.ndim == 1 and y_pred.size(1) > 1):
                y_pred = y_pred.softmax(dim=-1)

        if y_pred.ndim == 2 and y_pred.size(1) == 1:
            y_pred = y_pred[:, 0]
        elif y_pred.ndim == 2 and y_pred.size(1) == 2:
            y_pred = y_pred[:, 1]

        if y_true.ndim == 2 and y_true.size(1) == 1:
            y_true = y_true[:, 0]
        elif y_true.ndim == 2 and y_true.size(1) == 2:
            y_true = y_true[:, 1]

        return y_pred, y_true

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader):
        y_pred, y_true = self.predict(dataloader=dataloader)
        metrics = self.compute_metrics(y_pred, y_true)
        return metrics, y_pred, y_true

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


class PrematchTrainer(SupervisedClassificationTrainer):
    @torch.no_grad()
    def _predict(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(self.model, list):
            self.model = [model.eval().to(self.device) for model in self.model]
        else:
            self.model.eval()
            self.model.to(self.device)

        _match_ids, _pred, _true = [], [], []
        for batch in dataloader:
            batch = to_device(batch, self.device)
            if isinstance(batch, dict):
                x, y, mid = batch, batch['y'], batch['match_id']
            elif isinstance(batch, list):
                x, y = batch
                mid = x['match_id']
            else:
                raise Exception

            pred = self.forward(x)
            _match_ids.append(mid)
            _pred.append(pred)
            _true.append(y)

        match_ids = torch.cat(_match_ids).cpu()
        pred = torch.cat(_pred, dim=1).cpu()
        true = torch.cat(_true).cpu()
        return match_ids, pred, true

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        match_ids, y_pred, y_true = self._predict(dataloader=dataloader)
        y_pred = y_pred.mean(dim=0)

        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            y_pred = y_pred.sigmoid()
            
        elif isinstance(self.loss_fn, nn.CrossEntropyLoss):
            y_pred = y_pred.softmax(dim=1)

        else:
            if (y_true.ndim == 2 and y_true.size(1) == 1) or (y_true.ndim == 1 and y_pred.size(1) == 1):
                y_pred = y_pred.sigmoid()
            elif (y_true.ndim == 2 and y_true.size(1) > 1) or (y_true.ndim == 1 and y_pred.size(1) > 1):
                y_pred = y_pred.softmax(dim=-1)

        if y_pred.ndim == 2 and y_pred.size(1) == 1:
            y_pred = y_pred[:, 0]
        elif y_pred.ndim == 2 and y_pred.size(1) == 2:
            y_pred = y_pred[:, 1]

        if y_true.ndim == 2 and y_true.size(1) == 1:
            y_true = y_true[:, 0]
        elif y_true.ndim == 2 and y_true.size(1) == 2:
            y_true = y_true[:, 1]

        if match_ids.ndim == 2:
            match_ids = match_ids[:, 0]
            
        return match_ids, y_pred, y_true

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        match_ids, y_pred, y_true = self.predict(dataloader=dataloader)
        metrics = self.compute_metrics(y_pred, y_true)
        return metrics, match_ids, y_pred, y_true

