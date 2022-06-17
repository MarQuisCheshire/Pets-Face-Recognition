import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter


class FocalLoss(nn.Module):
    def __init__(self, num_class: int, gamma=0, eps=1e-7, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        self.adaptive_flag = bool(alpha)
        if self.adaptive_flag:
            self.alpha = Parameter(torch.Tensor(num_class))
            self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.adaptive_flag:
            init.ones_(self.alpha)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.adaptive_flag:
            input = self.alpha * input
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
