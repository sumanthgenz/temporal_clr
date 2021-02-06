import torch
import torch.nn as nn
import torchvision
import torchaudio
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
from tqdm import tqdm

import warnings
import glob
import gc 
import os
import socket

from augment import *
from metrics import *
from models import *
# from torchaudio_transforms import *

torchaudio.set_audio_backend("sox_io") 
os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")

data = ""
host = socket.gethostname()
if host == "stout":
    data = "big"
elif socket.gethostname() == "greybeard":
    data = "ssd"

class TemporalShuffleData(Dataset):

    def __init__(self, data_type):
        self.dataType = data_type
        self.dir = "/{}/kinetics_audio/{}".format(data, data_type)

        # 4! = 24, for shuffle order prediction
        self.num_classes = 24
        self.wav_paths = self.get_all_files()
        
    def get_all_files(self):
        wav_paths = []
        for path in glob.glob(f'{self.dir}/**/*.wav'):
            wav_paths.append(path)
        return wav_paths

    def get_pickle(self, classPath):
        with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
            result = pickle.load(handle)
        return result
    
    def __len__(self):
        return len(self.wav_paths)

    def getNumClasses(self):
        return self.num_classes

    def __getitem__(self, idx):
        try:
            filePath = self.wav_paths[idx]
            shuffle, label = get_temporal_shuffle(filePath)
            # if torch.sum(shuffle) == 0:
            #     label = 24
            return shuffle.numpy(), label

        except:
            # filePath = self.wav_paths[idx]
            # shuffle, label = get_temporal_shuffle(filePath)

            # return shuffle, label
            return None, None

class TemporalContrastiveData(Dataset):

    def __init__(self, data_type):
        self.dataType = data_type
        self.dir = "/{}/kinetics_audio/{}".format(data, data_type)
        self.wav_paths = self.get_all_files()
        
    def get_all_files(self):
        wav_paths = []
        for path in glob.glob(f'{self.dir}/**/*.wav'):
            wav_paths.append(path)
        return wav_paths

    def get_pickle(self, classPath):
        with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
            result = pickle.load(handle)
        return result
    
    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        try:
            filePath = self.wav_paths[idx]
            anchor, spatial = get_augmented_views(filePath)
            _, temporal, shuffle_label = get_temporal_shuffle(filePath)

            return anchor, spatial, temporal, torch.tensor([float(shuffle_label)])

        except:
            return torch.ones((128, 2040))/0, torch.ones((128, 2040))/0, torch.ones((128, 2040))/0, torch.tensor([float('inf')])

            # return torch.tensor([float('NaN')]), torch.tensor([float('NaN')]), torch.tensor([float('NaN')]), torch.tensor([float('NaN')])
            # return None, None, None, None
            # return torch.zeros(2, 128, 1000), torch.zeros(2, 128, 1000), torch.zeros(2, 128, 999), torch.zeros(1)

class TemporalPermutesDataset(Dataset):

    def __init__(self, data_type):
        self.dataType = data_type
        self.dir = "/{}/kinetics_audio/{}".format(data, data_type)
        self.wav_paths = self.get_all_files()
        
    def get_all_files(self):
        wav_paths = []
        for path in glob.glob(f'{self.dir}/**/*.wav'):
            wav_paths.append(path)
        return wav_paths

    def get_pickle(self, classPath):
        with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
            result = pickle.load(handle)
        return result
    
    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        try:
            filePath = self.wav_paths[idx]
            # num_label = int((filePath.split('/')[4]).split('_')[0]) - 1
            # wav, samp_freq = torchaudio.load(filePath)
            # feat = np.transpose(np.array(torchaudio.compliance.kaldi.mfcc(wav, sample_frequency=self.samp_freq)))
            # return feat, num_label, self.seq_len

            anchor, permutes = get_temporal_shuffle_views(filePath)
            # return view1.type(torch.FloatTensor), view2.type(torch.FloatTensor), t1, t2
            return anchor, permutes

        except:
            return None, None

class TemporalSupervised(Dataset):

    def __init__(self, data_type):
        self.dataType = data_type
        self.dir = "/{}/kinetics_audio/{}".format(data, data_type)
        self.wav_paths = self.get_all_files()
        
    def get_all_files(self):
        wav_paths = []
        for path in glob.glob(f'{self.dir}/**/*.wav'):
            wav_paths.append(path)
        return wav_paths

    def get_pickle(self, classPath):
        with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
            result = pickle.load(handle)
        return result
    
    def __len__(self):
        return len(self.wav_paths)

    def getNumClasses(self):
        return self.num_classes

    def __getitem__(self, idx):
        try:
            filePath = self.wav_paths[idx]
            spec, label = get_supervised_data(filePath)

            return spec.numpy(), label
        except:
            return None, None



if __name__ == '__main__':
    

    # temporal_shuffle_data = TemporalShuffleData("train")
    # before = 0
    # after = 0
    # skipped = 0
    # for i in tqdm(range(500, 1000)):
    #     encoder = TemporalOrderPrediction(num_classes=24)

    #     shuffle, label = temporal_shuffle_data.__getitem__(i)
    
    #     try:
    #         shuffle = torch.from_numpy(shuffle).type(torch.FloatTensor)
    #         if torch.mean(shuffle) == 0:
    #             before += 1
    #         shuffle = encoder((shuffle).unsqueeze(0))
    #         if torch.isnan(shuffle).any():
    #             after +=1 
    #     except:
    #         skipped +=1
    # print(before)
    # print(after)
    # print(skipped)

    data = TemporalContrastiveData("train")
    ax, sx, tx, lx = data.__getitem__(0)
    ay, sy, ty, ly = data.__getitem__(1)
    a, s, t, l = torch.stack((ax, ay)), torch.stack((sx, sy)), torch.stack((tx, ty)), torch.Tensor([lx, ly])
    print(a.shape)
    print(s.shape)
    print(t.shape)
    print(l.shape)

    model = TemporalContrastive(batch_size=2)

    batch = (a, s, t, l)
    print(model._encode_temporal(t).shape)
    
    output = model(batch)
    loss = model.loss(output)


    for i in list(output.values()):
        print(i)

    for k in list(loss.values()):
        print(k)

    # x = torch.rand(2, 128, 1000)
    # y = torch.zeros(2, 128, 1000)
    # print(x/0)

    # batch = torch.stack((x,y))
    # filtered =  torch.Tensor(list(filter(lambda x: x.sum().item()>0, batch)))

    # filtered = batch[batch.sum().item() != 0]

    # x, y = torch.tensor([float('NaN'), 1, 2]),  torch.tensor([float('inf'), float('inf'), float('inf')])
    # test = torch.stack((x,y))
    # filtered = test[~test.isinf()]
    # filtered = test[test == test]

    # print(test)
    # print(filtered)

    # output = model(data.__getitem__(0))

    # print(modules)


        # a = [np.array([1,2,3]), np.array([4,5,6])]
        # # a = [torch.Tensor([1,2,3]), torch.Tensor(4,5,6)]
        # b = torch.Tensor(a)
        # print(type(b[1]))


