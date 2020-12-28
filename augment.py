import torch
import torchvision
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import pickle
import tqdm
from tqdm import tqdm


import warnings
import glob

from metrics import *
from transforms import *

torchaudio.set_audio_backend("sox_io") 
os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")

def get_wave(path):
    wave, samp_freq = torchaudio.load(path)
    wave = wave.mean(dim=0) #avg both channels to get single audio strean
    return wave, samp_freq


def get_mfcc(wave, samp_freq=16000):
    return np.array((torchaudio.transforms.MFCC(sample_rate=samp_freq)(wave.unsqueeze(0))).mean(dim=0))


def get_mel_spec(wave, samp_freq=16000):
    wave = torch.unsqueeze(wave, 0)
    return (torchaudio.transforms.MelSpectrogram(sample_rate=samp_freq)(wave))[0,:,:]


def get_log_mel_spec(wave, samp_freq=16000):
    wave = torch.unsqueeze(wave, 0)
    spec = torchaudio.transforms.MelSpectrogram()(wave)
    return spec.log2()[0,:,:]


def augment(sample, wave_transform, spec_transform, threshold, fixed_crop=True):
    wave = wave_transform(threshold)(sample)
    wave = wave.type(torch.FloatTensor)
    spec = get_log_mel_spec(wave)

    #suppressing "assert mask_end - mask_start < mask_param" for time/freq masks
    # try:
    #     return spec_transform(threshold)(SpecFixedCrop(threshold)(spec[15:]))
    # except:
    #     # return SpecFixedCrop(threshold)(spec)
    #     return spec_transform(threshold)(SpecFixedCrop(threshold)(spec[15:]))


    if fixed_crop:
        spec = spec_transform(threshold)(SpecFixedCrop(threshold)(spec))
        spec[torch.isinf(spec)] = 0
        return spec
    
    spec = spec_transform(threshold)(SpecRandomCrop(threshold)(spec))
    spec[torch.isinf(spec)] = 0
    return spec

def get_augmented_views(path):
    sample, _ = get_wave(path)

    wave1 =  random.choice(list(wave_transforms.values()))
    spec1 =  random.choice(list(spec_transforms.values()))
    threshold1 = random.uniform(0.0, 0.5)

    wave2 =  random.choice(list(wave_transforms.values()))
    spec2 =  random.choice(list(spec_transforms.values()))
    threshold2 = random.uniform(0.0, 0.5)

    # wave1 = WaveIdentity
    # wave2 = WaveIdentity

    # spec1 = SpecShuffle
    # spec2 = SpecCheckerNoise

    print(wave1, spec1)
    print(wave2, spec2)

    return augment(sample, wave1, spec1, threshold1), augment(sample, wave2, spec2, threshold2), (wave1, spec1), (wave2, spec2)

def get_temporal_permutes(path):
    sample, _ = get_wave(path)
    wave = WaveIdentity
    spec1 = SpecIdentity
    spec2 = SpecPermutes
    threshold = random.uniform(0.0, 0.5)

    # Return (anchor, permutes), anchor is single sample, permutes is a list of samples
    return augment(sample, wave, spec1, threshold, fixed_crop=False), augment(sample, wave, spec2, threshold)
    
def get_temporal_shuffle(path):
    #assume num_segments = 4
    shuffle_idx = random.randint(0, 23)
    anchor, permutes = get_temporal_permutes(path)
    # return permutes[shuffle_idx], shuffle_idx
    return permutes[shuffle_idx], shuffle_idx


if __name__ == '__main__':
    for _ in tqdm(range(1)):
        filepath = "/{dir}/kinetics_audio/train/25_riding a bike/0->--JMdI8PKvsc.wav".format(dir = data)
        # view1, view2, _, _ = get_augmented_views(filepath)
        shuffle, label = get_temporal_shuffle(filepath)
 
    #     view1, view2 = permutes[5], permutes[10]
    
    # print(permutes.shape)
    # f = plt.figure()
    # f.add_subplot(1, 2, 1)
    # plt.imshow(view1)

    # f.add_subplot(1, 2, 2)
    # plt.imshow(view2)
    # plt.savefig("Desktop/log_mel_two_views.png")


