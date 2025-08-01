import gc
import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from transformers import AutoModel

from utils.model_utils import mean_pool, freeze_model, _print
from utils.training_utils import ScoreMatchingLoss, ScoreNormMetric, CosineWarmup


class TransportModule(pl.LightningModule):
    def __init__(self, config, score_model):
        super().__init__()
        self.config = config
        self.sigma = self.config.optim.sigma
        self.loss_fn = ScoreMatchingLoss(sigma=self.sigma)

        self.metrics = {
            "preds_norm": ScoreNormMetric(),
            "targets_norm": ScoreNormMetric(),
        }

        self.model = score_model
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
            z = mean_pool(embeds=embeddings, attention_mask=batch['attention_mask'])
        elif embeddings.ndim == 1:
            # Latents with correct dims are provided during langevin transport
            z = embeddings.unsqueeze(0)  # Only need to introduce a batch dimension
        else:
            raise ValueError(f"Incorrect embedding dim of {embeddings.ndim}")

        # Score matching loss computed on perturbed embeddings
        noise = torch.randn_like(z)
        z_prime = z + (noise * self.sigma)
        s_theta = self.model(z_prime)

        return s_theta, z, z_prime

    def compute_loss(self, batch, return_target=False):
        """Helper method to handle loss calculation"""
        score, z, z_prime = self.forward(batch)
        loss = self.loss_fn(score, z, z_prime).mean()
        if return_target:
            with torch.no_grad():
                target = (z - z_prime) / (self.sigma ** 2)
            return loss, score, target
        else:
            return loss, score

    
    # -------# Training / Evaluation #-------- #
    def on_train_start(self):
        self.loss_fn.to(self.device)

    def training_step(self, batch, batch_idx):
        train_loss, _ = self.compute_loss(batch)
        self.log(name="train/loss", value=train_loss.item(), on_step=True, on_epoch=False, logger=True, sync_dist=True)
        self.save_ckpt()
        return train_loss

    def on_validation_start(self):
        self.loss_fn.to(self.device)

    def validation_step(self, batch, batch_idx):
        val_loss, _ = self.compute_loss(batch)
        self.log(name="val/loss", value=val_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return val_loss

    def on_test_start(self):
        self.loss_fn.to(self.device)
        for metric in self.metrics.values():
            metric.to(self.device)

    def test_step(self, batch, batch_idx):
        test_loss, pred_score, target_score = self.compute_loss(batch, return_target=True)
        self.metrics['preds_norm'].update(pred_score)
        self.metrics['targets_norm'].update(target_score)
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

    @torch.no_grad()
    def compute_embedding(self, batch):
        outs = self.embed_model(**batch)
        embeds = outs.last_hidden_state
        return embeds


    # -------# Helper Functions #-------- #
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