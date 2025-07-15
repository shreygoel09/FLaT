import sys
import torch.nn as nn


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