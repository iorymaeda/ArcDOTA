from typing import Callable, Optional, Literal

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor


class OrdinalCrossEntropyLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super(OrdinalCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weights = torch.abs(target.argmax(dim=1) - input.argmax(dim=1)) / (input.size()[1] - 1)
        loss = (1 + weights) * F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction='none',
                               label_smoothing=self.label_smoothing)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

class BinaryHingeLoss(torch.nn.Module):
    def __init__(self, y_infimum: Literal[-1, 0], reduction='mean'):
        self.y_infimum = y_infimum
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.y_infimum == 0:
            target = target*2 - 1
            
        loss = torch.max(torch.zeros_like(target), 1 - target * input)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()