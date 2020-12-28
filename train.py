import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from supervised import *
from contrastive import *

modes = ["supervised", "contrastive", "linear"]
mode = modes[1]

if __name__ == "__main__":

  if mode == "supervised":

    model = SupervisedModel()
    wandb_logger.watch(model, log='gradients', log_freq=10)

    trainer = pl.Trainer(
        default_root_dir='/home/sgurram/Desktop/pcl_checkpoint', 
        gpus=2, 
        max_epochs=200, 
        logger=wandb_logger,
        accumulate_grad_batches=1, 
        distributed_backend='ddp')  

    trainer.fit(model)

  elif mode == "contrastive":

    model = ContrastiveModel()
    wandb_logger.watch(model, log='gradients', log_freq=10)

    trainer = pl.Trainer(
        default_root_dir='/home/sgurram/Desktop/pcl_contrastive/infonce', 
        gpus=2, 
        max_epochs=200, 
        logger=wandb_logger,
        accumulate_grad_batches=4, 
        distributed_backend='ddp')  

    trainer.fit(model)


  elif mode == "linear":

      model = LinearClassifier()
      wandb_logger.watch(model, log='gradients', log_freq=10)

      trainer = pl.Trainer(
          default_root_dir='/home/sgurram/Desktop/pcl_linear/infonce', 
          gpus=2, 
          max_epochs=200, 
          logger=wandb_logger,
          accumulate_grad_batches=1, 
          distributed_backend='ddp')  

      trainer.fit(model)
