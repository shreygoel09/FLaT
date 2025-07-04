import torch
import torch.nn as nn

class ScoreMatchingLoss(nn.Module):
    def __init__(self, config, device, sigma):
        self.config = config
        self.device = device
        self.sigma = sigma

    def forward(self, s_theta, z, z_prime):
        target = (z - z_prime) / (self.sigma ** 2)
        loss = ((s_theta - target) ** 2).sum(dim=1).mean()
        return loss
