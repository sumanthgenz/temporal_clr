import torch
import torchaudio
import torchvision
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet

import numpy as np
import pandas as pd 
import warnings
import glob
from tqdm import tqdm
import pickle
from collections import Counter
import copy
import os

from dataloader import *
from metrics import *


#Implementation from
# https://github.com/CVxTz/COLA_pytorch/blob/master/audio_encoder/encoder.py
class AudioFeatureModel(torch.nn.Module):
    def __init__(self, dropout=0.1, model_dimension=512):
        super(AudioFeatureModel, self).__init__()

        self._model_dimension = model_dimension
        self._dropout = 0.1

        self._cnn1 = torch.nn.Conv2d(
                                in_channels=1, 
                                out_channels=3, 
                                kernel_size=3)

        self._efficientnet = EfficientNet.from_name(
                                "efficientnet-b0", 
                                include_top=False, 
                                drop_connect_rate=self._dropout)

        self._fc1 = nn.Linear(1280, self._model_dimension)

        self._dropout = torch.nn.Dropout(p=self._dropout)

        self._layer_norm = torch.nn.LayerNorm(normalized_shape=self._model_dimension)


    def forward(self, input_audio):
        #Input B * C * M * T

        # x = x.type(torch.FloatTensor)

        #Filter out NaN -inf values at top of spec (mel bins), and unsqueeze channel
        x = input_audio.unsqueeze(1)

        x = self._cnn1(x)
        x = self._efficientnet(x)
        x =  x.squeeze(3).squeeze(2)
        x = self._dropout(self._fc1(x))
        x = self._dropout(torch.tanh(self._layer_norm(x)))

        #Output B * D, D=1024
        return x

#Implemenation from https://github.com/CannyLab/aai/blob/e51bc4f0926530c39f289a948e0a1daebed3475a/aai/research/gptcaptions/models/encoders/predictive_byol.py#L21
class VideoFeatureModel(torch.nn.Module):
    def __init__(self, dropout=0.1, model_dimension=128):
        super(VideoFeatureModel, self).__init__()

        self._model_dimension = model_dimension
        self._feature_dimension = model_dimension/4


        self._dropout = dropout

        self._resnet_model = torchvision.models.resnet18(pretrained=True)

        self._feature_model = torch.nn.Sequential(
            self._resnet_model.conv1,
            self._resnet_model.bn1,
            self._resnet_model.relu,
            self._resnet_model.maxpool,
            self._resnet_model.layer1,
            self._resnet_model.layer2,
            self._resnet_model.layer3,
            self._resnet_model.layer4,
        )

        self._frame_input_projection = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self._feature_dimension),
            torch.nn.Linear(self._model_dimension, self._feature_dimension),
            torch.nn.ReLU(),
        )

        self._encoder_layer = torch.nn.moduels.TransformerEncoderLayer(d_model=self._feature_dimension,
                                                                 nhead=self._num_heads,
                                                                 dim_feedforward=self._model_dimension,
                                                                 dropout=self._dropout,
                                                                 activation='relu')
        self._encoder = torch.nn.modules.TransformerEncoder(encoder_layer=self._encoder_layer,
                                                                    num_layers=self._num_layers)

        self._fc1 = nn.Linear(self._feature_dimension, self._model_dimension)


    #Implementation from https://github.com/CannyLab/aai/blob/e51bc4f0926530c39f289a948e0a1daebed3475a/aai/utils/torch/masking.py#L39
    # def sequence_mask(lengths, maxlen=None, right_aligned=False):
    #     # https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036
    #     if maxlen is None:
    #         maxlen = lengths.max()
    #     matrix = torch.unsqueeze(lengths, dim=-1)
    #     row_vector = torch.arange(0, maxlen, 1).type_as(matrix)
    #     if not right_aligned:
    #         mask = row_vector < matrix
    #     else:
    #         mask = row_vector > (-matrix + (maxlen - 1))

    #     return mask.bool()

    # def get_src_conditional_mask(max_sequence_length):
    #     mask = (torch.triu(torch.ones(max_sequence_length, max_sequence_length)) == 1).transpose(0, 1)
    #     return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))


    def forward(self, input_video):
        #Input B * T * H * W * C

        # x = x.type(torch.FloatTensor)

        video_frames = input_video.reshape(-1, *input_video.shape[2:])
        frames_encoded = self._feature_model(video_frames.contiguous())
        frames_encoded = frames_encoded.reshape(input_video.shape[0], -1,
                                                *frames_encoded.shape[1:]).mean(dim=(3, 4))

        #B * T * D
        frames_encoded = self._frame_input_projection(frames_encoded.reshape(-1, self._feature_dimension)).reshape(
            frames_encoded.shape[0], frames_encoded.shape[1], self._model_dimension)

        #B * D
        video_features = self._encoder(src=frames_encoded).transpose(0, 1).mean(dim=0)

        #B * F
        video_features = self._fc1(self._dropout(video_features))

        return video_features

class BYOLEncoder(torch.nn.Module):

    def __init__(self):
        super(BYOLEncoder, self).__init__()

        self._model_dimension = 512

        self._audio_feature_model = AudioFeatureModel(
                                dropout=0.1,
                                model_dimension=self._model_dimension)

        self._video_feature_model = VideoFeatureModel(
                                dropout=0.1,
                                model_dimension=self._model_dimension)


        #Implementation from
        #https://github.com/CannyLab/aai/blob/ddc76404bdfe15fb8218c31d9dc6859f3d5420db/aai/research/gptcaptions/models/encoders/predictive_byol.py
        self._representation_mlp = torch.nn.Sequential(
            torch.nn.Linear(self._model_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )
        self._byol_predictor = torch.nn.Sequential(
            torch.nn.Linear(self._model_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )

        self._translation_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * self._model_dimension, self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )

        self._target_encoder = None
        self._target_networks_initialized = False
        self._ema_beta = ema_beta


    def get_paramaters(self,):
        params = []
        params += list(self._encoder.parameters())
        if self._target_encoder is not None:
            params += list(self._target_encoder.parameters())
        params += list(self._representation_mlp.parameters())
        params += list(self._byol_predictor.parameters())
        params += list(self._translation_mlp.parameters())

        return params

    def reset_moving_average(self):
        del self._target_encoder
        self._target_encoder = None


    def _ema_copy_model(self, online_model, target_model):
        for current_params, target_params in zip(online_model.parameters(), target_model.parameters()):
            old_weight, new_weight = target_params.data, current_params.data
            target_params.data = old_weight * self._ema_beta + (1 - self._ema_beta) * new_weight


    def update_moving_average(self):
        if self._target_encoder is not None:
            self._ema_copy_model(self._encoder, self._target_encoder)


    def byol_encode(self, x, online=True):
        if online:
            x = self._encoder(x)
        else:
            if not self._target_networks_initialized:
                self._target_encoder = copy.deepcopy(self._encoder)
                self._target_networks_initialized = True
            x = self._target_encoder(x)
        x = self._representation_mlp(x)
        return x

    
    def forward(self, x):
        x1, x2 = x

        x1 = self._feature_model(x1)
        x2 = self._feature_model(x2)

        return x1, x2
    
