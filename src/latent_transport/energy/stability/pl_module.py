import gc
import torch
import torch.nn as nn
import lightning.pytorch as pl

from transformers import AutoModel
from torchmetrics import SpearmanCorrCoef, MeanSquaredError, R2Score, PearsonCorrCoef

from utils.model_utils import _print, mean_pool, freeze_model
from utils.training_utils import CosineWarmup


class TransportModule(pl.LightningModule):
    def __init__(self, config, energy_model):
        super().__init__()
        self.config = config
        self.loss_fn = nn.MSELoss(reduction='none')

        self.metrics = {
            "spearman": SpearmanCorrCoef(),
            "mse": MeanSquaredError(),
            "r2": R2Score(),
            "pearson": PearsonCorrCoef()
        }
        
        self.model = energy_model
        self.embed_model = AutoModel.from_pretrained(self.config.lm.pretrained_esm)
        freeze_model(self.embed_model)
        self.embed_model.eval()


    # -------# Classifier step #-------- #
    def forward(self, batch):
        if 'input_ids' in batch:
            embeddings = self.compute_embedding(batch)
        else:
            embeddings = batch['embeds']

        if embeddings.ndim == 3:
            # Mean pooling ignoring pad tokens
            pooled = mean_pool(embeds=embeddings, attention_mask=batch['attention_mask'])
        elif embeddings.ndim == 1:
            # Latents with correct dims are provided during langevin transport
            # Only need to introduce a batch dimension
            pooled = embeddings.unsqueeze(0)
        else:
            raise ValueError(f"Incorrect embedding dim of {embeddings.ndim} provided")
        
        logits = self.model(pooled)
        return logits

    
    # -------# Training / Evaluation #-------- #
    def training_step(self, batch, batch_idx):
        train_loss, _ = self.compute_loss(batch)
        self.log(name="train/loss", value=train_loss.item(), on_step=True, on_epoch=False, logger=True, sync_dist=True)
        self.save_ckpt()
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss, _ = self.compute_loss(batch)
        self.log(name="val/loss", value=val_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return val_loss

    def on_test_start(self):
        for metric in self.metrics.values():
            metric.to(self.device)

    def test_step(self, batch, batch_idx):
        test_loss, preds = self.compute_loss(batch)
        labels = batch['labels']

        for metric in self.metrics.values():
            metric.update(preds, labels)

        self.log("test/loss", test_loss, on_step=False, on_epoch=True, sync_dist=True)
        return test_loss

    def on_test_epoch_end(self):
        for name, metric in self.metrics.items():
            self.log(f'test/{name}', metric.compute(), sync_dist=True)
            metric.reset()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        path = self.config.training
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.optim.lr)
        lr_scheduler = CosineWarmup(
            optimizer,
            warmup_steps=path.warmup_steps,
            total_steps=path.max_steps,
        )
        scheduler_dict = {
            "scheduler": lr_scheduler,
            "interval": 'step',
            'frequency': 1,
            'monitor': 'val/loss',
            'name': 'learning_rate'
        }
        return [optimizer], [scheduler_dict]


    # -------# Loss and Test Set Metrics #-------- #
    def compute_loss(self, batch):
        """Helper method to handle loss calculation"""
        labels = batch['labels']
        preds = self.forward(batch)
        _print(f'logits: {preds}')
        _print(f'labels: {labels}')
        loss = self.loss_fn(preds, labels).mean()
        return loss, preds


    # -------# Helper Functions #-------- #
    @torch.no_grad()
    def compute_embedding(self, batch):
        outs = self.embed_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        embeds = outs.last_hidden_state
        return embeds

    def save_ckpt(self):
        curr_step = self.global_step
        save_every = self.config.checkpointing.save_every_n_steps
        if curr_step % save_every == 0 and curr_step > 0: 
            ckpt_path = f"{self.config.checkpointing.save_dir}/step={curr_step}.ckpt"
            self.trainer.save_checkpoint(ckpt_path)

    def get_state_dict(self, ckpt_path):
        """Helper method to load and process a trained model's state dict from saved checkpoint"""
        def remove_model_prefix(state_dict):
            for k in state_dict.keys():
                if "model." in k:
                    k.replace('model.', '')
            return state_dict  

        checkpoint = torch.load(ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = checkpoint.get("state_dict", checkpoint)

        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = remove_model_prefix(state_dict)
        
        return state_dict