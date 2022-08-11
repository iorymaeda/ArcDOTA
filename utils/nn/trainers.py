import torch
import torch.nn as nn
from torch.utils.data import DataLoader


loss_fns = nn.BCEWithLogitsLoss | nn.CrossEntropyLoss

class PremtachTrainer:
    def __init__(self, 
        model, loss_fn, optimizer, sheduler=None,
        metric=None, device='cuda', **kwargs):


        self.model: nn.Module = model.to(device)
        self.device = device
        self.metric = metric
        
        self.loss_fn: loss_fns = loss_fn
        self.optimizer: torch.optim.Optimizer = optimizer   
        self.sheduler:torch.optim.lr_scheduler._LRScheduler = sheduler
        
        
    def checkpoint(self) -> dict:
        cpoint =  {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        return cpoint
    
        
    def train_epoch(self, dataset: DataLoader) -> float:
        self.model.train()
        self.model.to(self.device)

        running_loss = 0
        for idx, batch in enumerate(dataset):
            running_loss+= self.train_step(batch)

        if self.sheduler: self.sheduler.step()
        return running_loss/idx
            

    def train_step(self, batch: dict[torch.Tensor|dict, torch.Tensor|dict]):
        batch = self.batch_to_device(batch, self.device)
        X, Y = batch, batch['y']
        
        outputs: torch.Tensor = self.model(X)
        loss: torch.Tensor = self.loss_fn(outputs, Y)    

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.metric:
            self.metric(outputs.sigmoid(), Y.int())
        
        return loss.item()


    def evaluate(self, dataset) -> list[torch.Tensor]:
        self.model.eval()
        self.model.to(self.device)

        val_pred, val_true = [], []
        with torch.inference_mode():
            for batch in dataset:
                batch = self.batch_to_device(batch, self.device)
                X, Y = batch, batch['y']
                pred: torch.Tensor = self.model(X)
                
                val_pred+= [pred.cpu()]
                val_true+= [Y.cpu()]
                
        return torch.cat(val_pred), torch.cat(val_true)


    @staticmethod
    def batch_to_device(batch: dict[torch.Tensor|dict, torch.Tensor|dict], device) \
        -> dict[torch.Tensor|dict, torch.Tensor|dict]:

        def recurssive(batch: dict):
            for k, v in batch.items():
                if isinstance(v, dict):
                    batch[k] = recurssive(v)

                elif isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            return batch

        return recurssive(batch)