import gc
import torch
import torch.nn as nn
import lightning.pytorch as pl

from omegaconf import OmegaConf
from transformers import AutoModel
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from torchmetrics import PearsonCorrCoef

from utils.model_utils import CosineWarmup, _print


class TransportModule(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        self.auroc = BinaryAUROC()
        self.accuracy = BinaryAccuracy()
        self.pearson = PearsonCorrCoef()
        
        self.esm_model = AutoModel.from_pretrained(self.config.lm.pretrained_esm)
        for p in self.esm_model.parameters():
            p.requires_grad = False

        self.model = model

    # -------# Classifier step #-------- #
    def forward(self, batch_or_tensor):
        if isinstance(batch_or_tensor, dict):
            if 'input_ids' in batch_or_tensor:
                embeddings = self.get_esm_embeddings(batch_or_tensor['input_ids'], batch_or_tensor['attention_mask'])
            else:
                embeddings = batch_or_tensor['embeds']
        else:
            embeddings = batch_or_tensor  # direct tensor input

        if embeddings.ndim == 3:
            pooled = embeddings.mean(dim=1) # Mean pool needed during training
        elif embeddings.ndim == 1:
             # During langevin transport, a latent w/ correct dimension is provided
            pooled = embeddings.unsqueeze(0)  # introduce a batch dimension
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

    def test_step(self, batch):
        test_loss, preds = self.compute_loss(batch)
        auroc, accuracy, pearson = self.get_metrics(batch, preds)
        self.log(name="test/loss", value=test_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log(name="test/AUROC", value=auroc.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log(name="test/accuracy", value=accuracy.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log(name="test/pearson", value=pearson.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return test_loss

    def on_test_epoch_end(self):
        self.auroc.reset()
        self.accuracy.reset()
    
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
    
    def save_ckpt(self):
        curr_step = self.global_step
        save_every = self.config.training.val_check_interval
        if curr_step % save_every == 0 and curr_step > 0: 
            ckpt_path = f"{self.config.checkpointing.save_dir}/step={curr_step}.ckpt"
            self.trainer.save_checkpoint(ckpt_path)
    

    # -------# Loss and Test Set Metrics #-------- #
    @torch.no_grad
    def get_esm_embeddings(self, input_ids, attention_mask):
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        return embeddings

    def compute_loss(self, batch):
        """Helper method to handle loss calculation"""
        labels = batch['labels']
        preds = self.forward(batch)
        _print(f'preds: {preds}')
        _print(f'probs: {torch.sigmoid(preds)}')
        loss = self.loss_fn(preds, labels).mean()
        return loss, preds

    def get_metrics(self, batch, preds):
        """Helper method to compute metrics"""
        labels = batch['labels']
        auroc = self.auroc.forward(preds, labels)
        accuracy = self.accuracy.forward(preds, labels)
        pearson = self.pearson.forward(torch.sigmoid(preds), labels)
        return auroc, accuracy, pearson


    # -------# Helper Functions #-------- #
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