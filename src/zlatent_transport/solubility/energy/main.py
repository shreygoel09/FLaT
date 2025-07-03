#!/usr/bin/env python3

import os
import wandb
import lightning.pytorch as pl

from omegaconf import OmegaConf
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.latent_transport.solubility.energy.solubility_module import SolubilityClassifier
from src.latent_transport.solubility.energy.dataloader import SolubilityDataModule, get_datasets


config = OmegaConf.load("/home/a03-sgoel/FLaT/src/configs/solubility.yaml")
wandb.login(key='2b76a2fa2c1cdfddc5f443602c17b011fefb0a8f')

# data
datasets = get_datasets(config)
data_module = SolubilityDataModule(
    config=config,
    train_dataset=datasets['train'],
    val_dataset=datasets['val'],
    test_dataset=datasets['test'],
)

# wandb logging
wandb.init(project=config.wandb.project, name=config.wandb.name)
wandb_logger = WandbLogger(**config.wandb)

# lightning checkpoints
lr_monitor = LearningRateMonitor(logging_interval="step")
checkpoint_callback = ModelCheckpoint(
    monitor="val/loss",
    save_top_k=1,
    mode="min",
    dirpath=config.checkpointing.save_dir,
    filename="best_model",
)

# lightning trainer
trainer = pl.Trainer(
    max_steps=config.training.max_steps,
    accelerator="cuda",
    devices=config.training.devices if config.training.mode=='train' else [0],
    strategy=DDPStrategy(find_unused_parameters=True),
    callbacks=[checkpoint_callback, lr_monitor],
    logger=wandb_logger,
    log_every_n_steps=config.training.log_every_n_steps,
    val_check_interval=config.training.val_check_interval
)

# Folder to save checkpoints
ckpt_dir = config.checkpointing.save_dir
os.makedirs(ckpt_dir, exist_ok=True)

# instantiate model
model = SolubilityClassifier(config)

# train or evalute the model
if config.training.mode == "train":
    trainer.fit(model, datamodule=data_module)

elif config.training.mode == "test":
    ckpt_path = os.path.join(ckpt_dir, "best_model.ckpt")
    state_dict = model.get_state_dict(ckpt_path)
    model.load_state_dict(state_dict)
    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
else:
    raise ValueError(f"{config.training.mode} is invalid. Must be 'train' or 'test'")

wandb.finish()
