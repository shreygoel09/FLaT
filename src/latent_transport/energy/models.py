import torch
import torch.nn as nn



class SolubilityClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.d_model,
            nhead=config.model.num_heads,
            dropout=config.model.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.model.num_layers)
        self.layer_norm = nn.LayerNorm(config.model.d_model)
        self.dropout = nn.Dropout(config.model.dropout)
        # self.mlp = nn.Sequential(
        #     nn.Linear(config.model.d_model, config.model.d_model // 2),
        #     nn.LayerNorm(config.model.d_model // 2),
        #     nn.ReLU(),
        #     nn.Dropout(config.model.dropout),
        #     nn.Linear(config.model.d_model // 2, 1),
        # )

        self.mlp = nn.Sequential(
            nn.Linear(config.model.d_model, config.model.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.d_model // 2, 1),
        )

    def forward(self, esm_embeds, batch):
        encodings = self.encoder(esm_embeds, src_key_padding_mask=(batch['attention_mask'] == 0))
        encodings = self.dropout(self.layer_norm(encodings))

        # Masked mean pooling over the sequence dimension
        mask = batch['attention_mask'].unsqueeze(-1)  # shape: [B, L, 1]
        masked_encodings = encodings * mask  # zero out padding tokens
        sum_encodings = masked_encodings.sum(dim=1)  # sum over tokens
        lengths = mask.sum(dim=1).clamp(min=1e-6)     # avoid division by zero
        pooled = sum_encodings / lengths              # mean over valid tokens

        # Apply MLP to pooled sequence-level representation
        logits = self.mlp(pooled).squeeze(-1)  # shape: [B]
        return logits


class StabilityRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.d_model,
            nhead=config.model.num_heads,
            dropout=config.model.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.model.num_layers)
        self.layer_norm = nn.LayerNorm(config.model.d_model)
        self.dropout = nn.Dropout(config.model.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(config.model.d_model, config.model.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.d_model // 2, 1),
        )

    def forward(self, esm_embeds, batch):
        encodings = self.encoder(esm_embeds, src_key_padding_mask=(batch['attention_mask'] == 0))
        encodings = self.dropout(self.layer_norm(encodings))

        # Masked mean pooling over the sequence dimension
        mask = batch['attention_mask'].unsqueeze(-1)  # shape: [B, L, 1]
        masked_encodings = encodings * mask  # zero out padding tokens
        sum_encodings = masked_encodings.sum(dim=1)  # sum over tokens
        lengths = mask.sum(dim=1).clamp(min=1e-6)     # avoid division by zero
        pooled = sum_encodings / lengths              # mean over valid tokens

        # Apply MLP to pooled sequence-level representation
        logits = self.mlp(pooled).squeeze(-1)  # shape: [B]
        return logits

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
    