import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from efficientnet_pytorch import EfficientNet

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


import numpy as np
import pandas as pd 
import warnings
import glob
from tqdm import tqdm
import pickle
from collections import Counter
import copy
import os

from metrics import *
from data import *

wandb_logger = WandbLogger(name='shuffle_prediction',project='temporal_contastive_learning')

#Implementation from
# https://github.com/CVxTz/COLA_pytorch/blob/master/audio_encoder/encoder.py
class TemporalOrderPrediction(pl.LightningModule):
    def __init__(self, dropout=0.1, model_dimension=512, num_classes=25):
        super(TemporalOrderPrediction, self).__init__()

        self._num_classes = num_classes

        #bsz*accum_grad = 16*64 = 1024 for effective batch_size
        self._bsz = 8

        self._model_dimension = model_dimension

        self._cnn1 = torch.nn.Conv2d(
                                in_channels=1, 
                                out_channels=3, 
                                kernel_size=3)

        self._efficientnet = EfficientNet.from_name(
                                "efficientnet-b0", 
                                include_top=False, 
                                drop_connect_rate=0.1)

        self._fc1 = nn.Linear(1280, self._model_dimension)
        self._fc2 = nn.Linear(self._model_dimension, self._num_classes)

        self._layer_norm1 = torch.nn.LayerNorm(normalized_shape=self._model_dimension)
        self._layer_norm2 = torch.nn.LayerNorm(normalized_shape=self._num_classes)

        self._dropout = torch.nn.Dropout(p=0.1)
        self._softmax = torch.nn.Softmax()

        self._lr = 1e-4
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, input):
        #Input B * C * M * T

        # x = x.type(torch.FloatTensor)

        #Filter out NaN -inf values at top of spec (mel bins), and unsqueeze channel
        x = input.unsqueeze(1)

        x = self._cnn1(x)
        x = self._efficientnet(x)
        x =  x.squeeze(3).squeeze(2)
        x = self._dropout(self._fc1(x))
        x = self._dropout(torch.tanh(self._layer_norm1(x)))
        x = self._softmax(self._fc2(x))

        #Output B * N, N = num_classes
        return x


    def training_step(self, batch, batch_idx):
        sample, label = batch
        logits = self.forward(sample)
        loss = self._loss(logits, label)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
      sample, label = batch
      logits = self.forward(sample)
      loss = self._loss(logits, label)

      top_1_accuracy = compute_accuracy(logits, label, top_k=1)
      top_5_accuracy = compute_accuracy(logits, label, top_k=5)
      
      logs = {
            'val_loss': loss,
            'val_top_1': top_1_accuracy,
            'val_top_5': top_5_accuracy}

      return logs

    def test_step(self, batch, batch_idx):
      sample, label = batch
      logits = self.forward(sample)
      loss = self._loss(logits, label)

      top_1_accuracy = compute_accuracy(logits, label, top_k=1)
      top_5_accuracy = compute_accuracy(logits, label, top_k=5)

      logs = {
            'test_loss': loss,
            'test_top_1': top_1_accuracy,
            'test_top_5': top_5_accuracy}

      return logs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([m['val_loss'] for m in outputs]).mean()
        avg_top1 = torch.stack([m['val_top_1'] for m in outputs]).mean()
        avg_top5 = torch.stack([m['val_top_5'] for m in outputs]).mean()

        logs = {
        'val_loss': avg_loss,
        'val_top_1': avg_top1,
        'val_top_5': avg_top5}

        return {'val_loss': avg_loss, 'log': logs}

    def collate_fn(self, batch):

        batch = np.transpose(batch)
        # print(batch[0].shape)
        # print(batch[1].shape)

        data =  torch.Tensor(list(filter(lambda x: x is not None, batch[0])))
        labels = torch.Tensor(list(filter(lambda x: x is not None, batch[1]))).long()
        return data, labels

    def train_dataloader(self):
        dataset = TemporalShuffleData(data_type='train')
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=self._bsz,
                                shuffle=True,
                                collate_fn=self.collate_fn,
                                num_workers=8)

    def val_dataloader(self):
          dataset = TemporalShuffleData(data_type='validate')
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=self._bsz,
                                  shuffle=True,
                                  collate_fn=self.collate_fn,
                                  num_workers=8)

    def test_dataloader(self):
        dataset = TemporalShuffleData(data_type='test')
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=self._bsz,
                                collate_fn=self.collate_fn,
                                shuffle=False,
                                num_workers=8)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)


class LinearClassifier(pl.LightningModule):

    def __init__(self):
        super(LinearClassifier, self).__init__()

        self.num_classes = 700
        self.fc1 = nn.Linear(1280, self.num_classes)
        self.lr = 1e-3
        self.loss = torch.nn.CrossEntropyLoss()
        self.bsz = 8

        self.path = 'Desktop/temporal_order_ckpt/temporal_contastive_learning/8cusj0o8/checkpoints/epoch=88.ckpt'
        self.base_model = TemporalOrderPrediction()
        # modules = list(self.base_model.children())[:]
        self.model = nn.Sequential(
                self.base_model._cnn1,
                self.base_model._efficientnet,
        )
        self.model.load_state_dict(torch.load(self.path), strict=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        with torch.no_grad():
            x = self.model(x)
        x =  x.squeeze(3).squeeze(2)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        sample, label = batch
        logits = self.forward(sample)
        loss = self.loss(logits, label)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
      sample, label = batch
      logits = self.forward(sample)
      loss = self.loss(logits, label)

      top_1_accuracy = compute_accuracy(logits, label, top_k=1)
      top_5_accuracy = compute_accuracy(logits, label, top_k=5)
      
      logs = {
            'val_loss': loss,
            'val_top_1': top_1_accuracy,
            'val_top_5': top_5_accuracy}

      return logs

    def test_step(self, batch, batch_idx):
      sample, label = batch
      logits = self.forward(sample)
      loss = self.loss(logits, label)

      top_1_accuracy = compute_accuracy(logits, label, top_k=1)
      top_5_accuracy = compute_accuracy(logits, label, top_k=5)

      logs = {
            'test_loss': loss,
            'test_top_1': top_1_accuracy,
            'test_top_5': top_5_accuracy}

      return logs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([m['val_loss'] for m in outputs]).mean()
        avg_top1 = torch.stack([m['val_top_1'] for m in outputs]).mean()
        avg_top5 = torch.stack([m['val_top_5'] for m in outputs]).mean()

        logs = {
        'val_loss': avg_loss,
        'val_top_1': avg_top1,
        'val_top_5': avg_top5}

        return {'val_loss': avg_loss, 'log': logs}

    def collate_fn(self, batch):

        batch = np.transpose(batch)
        # print(batch[0].shape)
        # print(batch[1].shape)

        data =  torch.Tensor(list(filter(lambda x: x is not None, batch[0])))
        labels = torch.Tensor(list(filter(lambda x: x is not None, batch[1]))).long()
        return data, labels

    def train_dataloader(self):
        dataset = TemporalSupervised(data_type='train')
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=self.bsz,
                                shuffle=True,
                                collate_fn=self.collate_fn,
                                num_workers=8)

    def val_dataloader(self):
          dataset = TemporalSupervised(data_type='validate')
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=self.bsz,
                                  shuffle=True,
                                  collate_fn=self.collate_fn,
                                  num_workers=8)


    def test_dataloader(self):
          dataset = TemporalSupervised(data_type='test')
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=self.bsz,
                                  shuffle=True,
                                  collate_fn=self.collate_fn,
                                  num_workers=8)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    
        #to replicate supcon cross-entropy, use these hparams: https://github.com/google-research/google-research/blob/master/supcon/scripts/cross_entropy_cifar10_resnet50.sh
        # optimizer = torch.optim.SGD(
        #                     self.parameters(), 
        #                     lr=self.lr, 
        #                     momentum=0.0, 
        #                     weight_decay=0,
        #                     nesterov=False)
        # return optimizer


        # scheduler = torch.optim.lr_scheduler.StepLR(
        #                     optimizer, 
        #                     step_size=40, 
        #                     gamma=0.1)
        # return [optimizer], [scheduler]
