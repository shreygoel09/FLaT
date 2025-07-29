import torch
import torch.nn as nn


class StabilityRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = self.config.model.d_model

        self.model = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),

            nn.Linear(d_model * 2, d_model),
            nn.GELU(),

            nn.Linear(d_model, d_model // 2),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)


class PermeabiltyRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = self.config.model.d_model

        self.model = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),

            nn.Linear(d_model * 2, d_model),
            nn.GELU(),

            nn.Linear(d_model, d_model // 2),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)
    

class SolubilityClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        d_model = self.config.model.d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

        # self.mlp = nn.Sequential(
        #     nn.Linear(config.model.d_model, config.model.d_model),
        #     nn.GELU(),
        #     nn.Dropout(config.model.dropout),

        #     nn.Linear(config.model.d_model, config.model.d_model // 2),
        #     nn.GELU(),
        #     nn.Dropout(config.model.dropout),

        #     nn.Linear(config.model.d_model // 2, config.model.d_model // 4),
        #     nn.GELU(),
        #     nn.Dropout(config.model.dropout),

        #     nn.Linear(config.model.d_model // 4, 1),
        # )

        # self.mlp = nn.Sequential(
        #     nn.Linear(config.model.d_model, config.model.d_model // 2),
        #     nn.ReLU(),
        #     nn.Dropout(config.model.dropout),
        #     nn.Linear(config.model.d_model // 2, 1),
        # )

