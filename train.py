import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models import *

wandb_logger = WandbLogger(name='shuffle_prediction',project='temporal_contastive_learning')


modes = ["order_prediction", "contrastive", "linear"]
mode = modes[1]

if __name__ == "__main__":

  if mode == "order_prediction":

    model = TemporalOrderPrediction(num_classes=24)
    wandb_logger.watch(model, log='gradients', log_freq=1)
    # wandb_logger = None

    trainer = pl.Trainer(
        default_root_dir='/home/sgurram/Desktop/temporal_order_ckpt', 
        gpus=2, 
        max_epochs=200, 
        logger=wandb_logger,
        accumulate_grad_batches=64, 
        distributed_backend='ddp')  

    trainer.fit(model)

  elif mode == "contrastive":

    model = TemporalContrastive()
    # wandb_logger.watch(model, log='gradients', log_freq=10)
    wandb_logger = None

    trainer = pl.Trainer(
        default_root_dir='/home/sgurram/Desktop/temporal_contrastive_ckpt', 
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
          default_root_dir='/home/sgurram/Desktop/temporal_linear_ckpt', 
          gpus=2, 
          max_epochs=50, 
          logger=wandb_logger,
          accumulate_grad_batches=64, 
          distributed_backend='ddp')  

      trainer.fit(model)
