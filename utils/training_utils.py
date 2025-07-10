import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric


class ScoreMatchingLoss(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, s_theta, z, z_prime):
        target = (z - z_prime) / (self.sigma ** 2)
        loss = F.mse_loss(
            input=s_theta,
            target=target,
            reduction='none'
        ).sum(dim=1)

        _print(f'target norm: {target.norm(dim=1).mean().item():4f}')
        _print(f'pred norm: {s_theta.norm(dim=1).mean().item():4f}')
        _print(f'loss: {loss.mean().item():4f}')
        
        return loss


class ScoreNormMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_norms", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, scores):
        norms = scores.norm(p=2, dim=1)
        self.sum_norms += norms.sum()
        self.total += norms.numel()
    
    def compute(self):
        return self.sum_norms / self.total
    

class CosineWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_ratio=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_ratio = eta_ratio  # The ratio of minimum to maximum learning rate
        super(CosineWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        decayed_lr = (1 - self.eta_ratio) * cosine_decay + self.eta_ratio

        return [decayed_lr * base_lr for base_lr in self.base_lrs]