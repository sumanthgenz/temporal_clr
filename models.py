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

class AudioFeatureModel(torch.nn.Module):
    def __init__(self, 
                dropout=0.1, 
                model_dimension=512):

        super(AudioFeatureModel, self).__init__()

        self.mel_freq = 128
        self.model_dimension = 512
        self.time_stpes = 300

        #audio convnet 
        self.conv1 = torch.nn.Conv1d(
                    in_channels=self.mel_freq, 
                    out_channels=self.model_dimension, 
                    kernel_size=2, 
                    stride=2,
        )

        self.conv2 = torch.nn.Conv1d(
                    in_channels=self.model_dimension, 
                    out_channels=self.model_dimension, 
                    kernel_size=2,
                    stride=2,
        )

        self.pool1 = nn.MaxPool1d(
                kernel_size=2,
                stride=2,
        )

        self.drop = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm1d(num_features=self.model_dimension)
        self.ln = nn.LayerNorm(normalized_shape=(self.model_dimension, self.time_stpes))

        self.audio_conv = nn.Sequential(
                self.conv1,
                self.conv2,
                self.bn,
                self.relu,
                self.pool1,
                self.drop,
        )

    def forward(self, input_audio):
        #Input [N * C * T]
        x = self.audio_conv(input_audio.type(torch.FloatTensor).to(input_audio.device))
        x = torch.einsum('ndt->ntd', [x])

        #Output [N * T * D]
        return x

class TemporalContrastive(pl.LightningModule):
    def __init__(self, 
                num_permutes=5,
                num_types=2,
                model_dimension=128, 
                feat_dimension=512,
                seqlen=125,
                batch_size=32, 
                num_heads=4, 
                num_layers=4, 
                dropout=0.1,
                learning_rate=1e-3,
                type_loss_weight=0.2,
                order_loss_weight=0.3,
                contrastive_loss_weight=0.5):

        super(TemporalContrastive, self).__init__()

        self._num_permutes = num_permutes
        self._num_types = num_types
        self._model_dimension = model_dimension
        self._feature_dimension = feat_dimension
        self._seqlen = seqlen
        self._batch_size = batch_size
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._dropout = dropout
        self._learning_rate = learning_rate
        self._type_loss_weight = type_loss_weight
        self._order_loss_weight = order_loss_weight
        self._contrastive_loss_weight = contrastive_loss_weight

        self._type_loss = nn.BCELoss()
        self._order_loss = nn.CrossEntropyLoss()
        self._contrastive_loss = NCELoss


        self._audio_feature_model = AudioFeatureModel(
                                dropout= self._dropout,
                                model_dimension=self._feature_dimension)

        self._input_projection = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self._feature_dimension),
            torch.nn.Linear(self._feature_dimension, self._model_dimension),
            torch.nn.ReLU(),
        )

        self._encoder_layer = torch.nn.modules.TransformerEncoderLayer(d_model=self._model_dimension,
                                                                 nhead=self._num_heads,
                                                                 dim_feedforward=self._model_dimension,
                                                                 dropout=self._dropout,
                                                                 activation='relu')
        self._encoder = torch.nn.modules.TransformerEncoder(encoder_layer=self._encoder_layer,
                                                        num_layers=self._num_layers,)

        self._type_mlp = torch.nn.Sequential(
            torch.nn.Linear(self._model_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._num_types),
            torch.nn.Softmax(),
        )

        self._order_mlp = torch.nn.Sequential(
            torch.nn.Linear(self._model_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._num_permutes),
            torch.nn.Softmax(),
        )

        self._contrastive_mlp = torch.nn.Sequential(
            torch.nn.Linear(self._model_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )

    def _feature_project(self, x):
        return self._input_projection(x.reshape(-1, self._feature_dimension)).reshape(
            x.shape[0], x.shape[1], self._model_dimension)

    def _encode_sequence(self, x, mlp_name):
        if mlp_name == 'contrastive':
            mlp = self._contrastive_mlp
        elif mlp_name == 'order':
            mlp = self._order_mlp
        elif mlp_name == 'type':
            mlp = self._type_mlp
        else:
            pass

        x = self._audio_feature_model(x)
        x = self._feature_project(x)
        encoded = self._encoder(src=x,).mean(dim=1)
  
        #Input [N * D] to mlp
        encoded = mlp(encoded)

        return encoded

    def filter_nans(self, batch):
        anchor, spatial, temporal, labels = batch
        anchor =  anchor[~anchor.isinf()].reshape(-1, anchor.shape[1], anchor.shape[2])
        spatial =  spatial[~spatial.isinf()].reshape(-1, spatial.shape[1], spatial.shape[2])
        temporal =  temporal[~temporal.isinf()].reshape(-1, temporal.shape[1], temporal.shape[2])
        labels =  labels[~labels.isinf()]
        return anchor, spatial, temporal, labels

    def forward(self, batch):
        anchor, spatial, temporal, idx_ord = self.filter_nans(batch)
        # anchor, spatial, temporal, idx_ord = batch

        # print(anchor.shape)
        # print(spatial.shape)
        # print(temporal.shape)
        # print(idx_ord.shape)

        anchor_clr = self._encode_sequence(anchor, 'contrastive')
        spatial_clr = self._encode_sequence(spatial, 'contrastive')
        spatial_type = self._encode_sequence(spatial, 'type')
        temporal_type = self._encode_sequence(temporal, 'type')
        temporal_ord = self._encode_sequence(temporal, 'order')

        anchor_clr = torch.nn.functional.normalize(anchor_clr, p=2, dim=-1)
        spatial_clr = torch.nn.functional.normalize(spatial_clr, p=2, dim=-1)

        mlp_outputs = {
            'x_clr': anchor_clr,
            'y_clr': spatial_clr,
            'x_type': spatial_type,
            'y_type': temporal_type,
            'x_ord': temporal_ord,
            'y_ord': idx_ord,
        }

        return mlp_outputs

    def loss(self, out):
        #type loss is binary classification, where spatial and temporal are classes {0, 1}
        type_loss = self._type_loss(out['x_type'], torch.zeros(out['x_type'].shape).to(out['x_type'].device))
        type_loss += self._type_loss(out['y_type'], torch.ones(out['y_type'].shape).to(out['y_type'].device))
        order_loss = self._order_loss(out['x_ord'], out['y_ord'].long())
        contrastive_loss = self._contrastive_loss(out['x_clr'], out['y_clr'])

        total_loss = torch.zeros([]).cuda()
        total_loss += self._type_loss_weight * type_loss
        total_loss += self._order_loss_weight * order_loss
        total_loss += self._contrastive_loss_weight * contrastive_loss

        type_acc = compute_accuracy(out['x_type'], torch.zeros(out['x_type'].shape[0]).to(out['x_type'].device), top_k=1)
        type_acc += compute_accuracy(out['y_type'], torch.ones(out['y_type'].shape[0]).to(out['y_type'].device), top_k=1)
        type_acc *= 0.5

        order_acc = compute_accuracy(out['x_ord'], out['y_ord'], top_k=1)

        metrics = {
            'type_loss': type_loss,
            'order_loss': order_loss,
            'contrastive_loss': contrastive_loss,
            'total_loss': total_loss,
            'type_acc': type_acc,
            'order_acc': order_acc
        }

        return metrics


    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        metrics = self.loss(outputs)
        loss = metrics['total_loss']
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        metrics = self.loss(outputs)
        logs = {
            'val_type_loss': metrics['type_loss'],
            'val_order_loss': metrics['order_loss'],
            'val_contrastive_loss': metrics['contrastive_loss'],
            'val_total_loss': metrics['total_loss'],
            'val_type_acc': metrics['type_acc'],
            'val_order_acc': metrics['order_acc'],
        }

        return logs

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        metrics = self.loss(outputs)
        logs = {
            'test_type_loss': metrics['type_loss'],
            'test_order_loss': metrics['order_loss'],
            'test_contrastive_loss': metrics['contrastive_loss'],
            'test_total_loss': metrics['total_loss'],
            'test_type_acc': metrics['type_acc'],
            'test_order_acc': metrics['order_acc'],
        }

        return logs

    def validation_epoch_end(self, outputs):
        type_loss = torch.stack([m['val_type_loss'] for m in outputs]).mean()
        order_loss = torch.stack([m['val_order_loss'] for m in outputs]).mean()
        contrastive_loss = torch.stack([m['val_contrastive_loss'] for m in outputs]).mean()
        total_loss = torch.stack([m['val_total_loss'] for m in outputs]).mean()
        type_acc = torch.stack([m['val_type_acc'] for m in outputs]).mean()
        order_acc = torch.stack([m['val_order_acc'] for m in outputs]).mean()

        logs = {
            'val_type_loss': type_loss,
            'val_order_loss': order_loss,
            'val_contrastive_loss': contrastive_loss,
            'val_total_loss': total_loss,
            'val_type_acc': type_acc,
            'val_order_acc': order_acc,
        }


        return {'val_loss': total_loss, 'log': logs}

    def _collate_fn(self, batch):
        # batch = np.asarray(batch)
        # print(batch.shape)
        # print(batch[0].shape)
        # print(batch[0][0].shape)
        # print(batch[1].shape)
        # print(batch[1][0].shape)
        # print(batch[2].shape)
        # print(batch[2][0].shape)        
        # print(batch[3].shape)

        # samples =  torch.Tensor(list(filter(lambda x: x is not None, batch[0])))
        # labels =  torch.Tensor(list(filter(lambda x: x is not None, batch[1])))
        # return samples
        
        # anchor =  torch.Tensor(list(filter(lambda x: x is not None, np.reshape(batch[0], (batch[0].shape[0], -1)))))
        # spatial =  torch.Tensor(list(filter(lambda x: x is not None, np.reshape(batch[1], (batch[1].shape[0], -1)))))
        # temporal =  torch.Tensor(list(filter(lambda x: x is not None, np.reshape(batch[2], (batch[2].shape[0], -1)))))
        # labels = torch.Tensor(list(filter(lambda x: x is not None, np.reshape(batch[3], (batch[3].shape[0], -1)))))

        # anchor = torch.reshape(anchor, (-1, batch[0].shape[1], batch[0].shape[2]))
        # spatial = torch.reshape(spatial, (-1, batch[1].shape[1], batch[1].shape[2]))
        # temporal = torch.reshape(temporal, (-1, batch[2].shape[1], batch[2].shape[2]))
        # labels = torch.reshape(labels, (-1, batch[3].shape[1], batch[3].shape[2]))

        # anchor =  torch.Tensor(list(filter(lambda x: x is not None, batch[0])))
        # spatial =  torch.Tensor(list(filter(lambda x: x is not None, batch[1])))
        # temporal =  torch.Tensor(list(filter(lambda x: x is not None, batch[2])))
        # labels = torch.Tensor(list(filter(lambda x: x is not None, batch[3])))

        anchor =  batch[0][batch[0] == batch[0]]
        spatial =  batch[1][batch[1] == batch[1]]
        temporal =  batch[2][batch[2] == batch[2]]
        labels = batch[3][batch[3] == batch[3]]

        return anchor, spatial, temporal, labels
        # return batch[0], batch[1], batch[2], batch[3]

    def train_dataloader(self):
        dataset = TemporalContrastiveData(data_type='train')
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=self._batch_size,
                                shuffle=True,
                                num_workers=8,)

    def val_dataloader(self):
          dataset = TemporalContrastiveData(data_type='validate')
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=self._batch_size,
                                  shuffle=True,
                                  num_workers=8,)


    def test_dataloader(self):
        dataset = TemporalContrastiveData(data_type='test')
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=self._batch_size,
                                shuffle=False,
                                num_workers=8,
                                collate_fn=self.collate_fn,)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)




#Implementation from
# https://github.com/CVxTz/COLA_pytorch/blob/master/audio_encoder/encoder.py
class TemporalOrderPrediction(pl.LightningModule):
    def __init__(self, dropout=0.1, model_dimension=512, num_classes=5):
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
        self.batch_size = 8

        # self.path = 'Desktop/temporal_order_ckpt/temporal_contastive_learning/8cusj0o8/checkpoints/epoch=88.ckpt'
        self.path = 'Desktop/epoch=88.ckpt'
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
                                batch_size=self.batch_size,
                                shuffle=True,
                                collate_fn=self.collate_fn,
                                num_workers=8)

    def val_dataloader(self):
          dataset = TemporalSupervised(data_type='validate')
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  collate_fn=self.collate_fn,
                                  num_workers=8)


    def test_dataloader(self):
          dataset = TemporalSupervised(data_type='test')
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=self.batch_size,
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
