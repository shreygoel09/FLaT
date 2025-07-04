#!/usr/bin/env python3

import os
import wandb
import lightning.pytorch as pl
import torch.nn as nn

from importlib import import_module
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from torchmetrics import SpearmanCorrCoef, MeanSquaredError
from torchmetrics.classification import (
    BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
)

from src.latent_transport.energy.pl_module import TransportModule
from src.latent_transport.energy.dataloader import CustomDataset, CustomDataModule
from src.latent_transport.energy.peptide_utils.tokenizer import SMILES_SPE_Tokenizer


# -------- Setup -------- #
mode = "energy"
prop = "perm"   # sol / perm

config = OmegaConf.load(f"/home/a03-sgoel/FLaT/src/configs/{mode}/{prop}.yaml")


# -------- Model Loader -------- #
energy_model_path = "src.latent_transport.energy.models"
tokenizer_path = "/home/a03-sgoel/FLaT/src/latent_transport/energy/peptide_utils"
registry = {
    "sol": {
        "energy_model": (energy_model_path, "SolubilityClassifier"),
        "metrics": {
            "auroc": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "f1": BinaryF1Score(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall()
        },
        "embedding_model": AutoModel.from_pretrained(config.lm.pretrained_esm),
        "tokenizer": AutoTokenizer.from_pretrained(config.lm.pretrained_esm),
        "loss_fn": nn.BCEWithLogitsLoss(reduction='none')
    },

    "perm": {
        "energy_model": (energy_model_path, "PermeabilityRegressor"),
        "metrics": {
            "spearman": SpearmanCorrCoef(),
            "mse": MeanSquaredError()
        },
        "embedding_model":  AutoModel.from_pretrained("aaronfeller/PeptideCLM-23M-all"),
        "tokenizer": SMILES_SPE_Tokenizer(f"{tokenizer_path}/new_vocab.txt", f"{tokenizer_path}/new_splits.txt"),
        "loss_fn": nn.MSELoss(reduction='none')
    }
}

property_dct = registry[prop]

# Load models
module_path, class_name = property_dct["energy_model"]
module = import_module(module_path)
ModelClass = getattr(module, class_name)
energy_model = ModelClass(config)

metrics = property_dct["metrics"]
loss_fn = property_dct["loss_fn"]
tokenizer = property_dct["tokenizer"]
embedding_model = property_dct["embedding_model"].eval()
for p in embedding_model.parameters(): p.requires_grad = False

# Load lightning module
pl_module = TransportModule(config, prop, energy_model, embedding_model, metrics, loss_fn)


# -------- Datasets -------- #
train_dataset = CustomDataset(config, config.data.train, prop, tokenizer)
val_dataset   = CustomDataset(config, config.data.val, prop, tokenizer)
test_dataset  = CustomDataset(config, config.data.test, prop, tokenizer)

data_module = CustomDataModule(
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset
)


# -------- Logging -------- #
wandb.init(project=config.wandb.project, name=config.wandb.name)
wandb_logger = WandbLogger(**config.wandb)


# -------- Callbacks -------- #
lr_monitor = LearningRateMonitor(logging_interval="step")
checkpoint_callback = ModelCheckpoint(
    monitor="val/loss",
    save_top_k=1,
    mode="min",
    dirpath=config.checkpointing.save_dir,
    filename="best_model",
)


# -------- Trainer -------- #
trainer = pl.Trainer(
    max_steps=config.training.max_steps,
    accelerator="cuda",
    devices=config.training.devices if config.training.mode == 'train' else [0],
    strategy=DDPStrategy(find_unused_parameters=True),
    callbacks=[checkpoint_callback, lr_monitor],
    logger=wandb_logger,
    log_every_n_steps=config.training.log_every_n_steps,
    val_check_interval=config.training.val_check_interval
)


# -------- Run -------- #
os.makedirs(config.checkpointing.save_dir, exist_ok=True)

if config.training.mode == "train":
    trainer.fit(pl_module, datamodule=data_module)
elif config.training.mode == "test":
    ckpt_path = os.path.join(config.checkpointing.save_dir, "best_model.ckpt")
    state_dict = pl_module.get_state_dict(ckpt_path)
    pl_module.load_state_dict(state_dict)
    trainer.test(pl_module, datamodule=data_module, ckpt_path=ckpt_path)
else:
    raise ValueError(f"{config.training.mode} is invalid. Must be 'train' or 'test'")

wandb.finish()
