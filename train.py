import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from encoder import *

modes = ["order_prediction", "contrastive", "linear"]
mode = modes[0]

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

    # model = ContrastiveModel()
    # wandb_logger.watch(model, log='gradients', log_freq=10)

    # trainer = pl.Trainer(
    #     default_root_dir='/home/sgurram/Desktop/pcl_contrastive/infonce', 
    #     gpus=2, 
    #     max_epochs=200, 
    #     logger=wandb_logger,
    #     accumulate_grad_batches=4, 
    #     distributed_backend='ddp')  

    # trainer.fit(model)
    pass


  elif mode == "linear":

      # model = LinearClassifier()
      # wandb_logger.watch(model, log='gradients', log_freq=10)

      # trainer = pl.Trainer(
      #     default_root_dir='/home/sgurram/Desktop/pcl_linear/infonce', 
      #     gpus=2, 
      #     max_epochs=200, 
      #     logger=wandb_logger,
      #     accumulate_grad_batches=1, 
      #     distributed_backend='ddp')  

      # trainer.fit(model)
      pass