#!/usr/bin/env python3

import os
import wandb
import lightning.pytorch as pl

from importlib import import_module
from omegaconf import OmegaConf
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import AutoTokenizer

from src.latent_transport.energy.models import StabilityRegressor
from src.latent_transport.energy.stability.pl_module import TransportModule
from src.latent_transport.energy.stability.dataloader import CustomDataset, CustomDataModule


config = OmegaConf.load(f"/home/a03-sgoel/FLaT/src/configs/energy/stability.yaml")


# -------- Model Loader -------- #
tokenizer = AutoTokenizer.from_pretrained(config.lm.pretrained_esm)
energy_model = StabilityRegressor(config)
pl_module = TransportModule(config, energy_model)


# -------- Datasets -------- #
train_dataset = CustomDataset(config, config.data.train, tokenizer)
val_dataset   = CustomDataset(config, config.data.val, tokenizer)
test_dataset  = CustomDataset(config, config.data.test, tokenizer)

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
