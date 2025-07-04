import sys
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

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


def mean_pool(embeds, attention_mask):
    assert embeds.ndim == 3  # B, L, D
    assert attention_mask.ndim == 2  # B, L

    mask = attention_mask.unsqueeze(-1).float()  # B, L, 1
    masked_embeds = embeds * mask
    summed = masked_embeds.sum(dim=1)  # B, D
    count = mask.sum(dim=1).clamp(min=1e-5)  # B, 1
    pooled = summed / count
    return pooled  # B, D


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False

def _print(text):
    print(text)
    sys.stdout.flush()