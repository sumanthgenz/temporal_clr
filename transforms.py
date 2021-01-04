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
from itertools import permutations  
from random import shuffle


import warnings
import glob
import socket


from metrics import *

assert torch.__version__.startswith("1.7")
assert torchaudio.__version__.startswith("0.7")

torchaudio.set_audio_backend("sox_io") 

data = ""
host = socket.gethostname()
if host == "stout":
    data = "big"
elif socket.gethostname() == "greybeard":
    data = "ssd"

class WaveIdentity():

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, wave):
        return wave


class WaveSegment():

    def __init__(self, threshold):
        self.segment_size = 1 - threshold

    def __call__(self, wave):
        size = int(wave.shape[0] * self.segment_size)
        start = random.randint(0, (wave.shape[0] - size))
        return wave[start : (start + size)]


class WaveGaussianNoise():

    def __init__(self, threshold):
        self.noise_intensity = threshold
        self.constant = 0.2

    def __call__(self, wave):
        noise = self.noise_intensity * self.constant
        return wave + (noise * np.random.normal(size=wave.shape[0]))


class WaveAmplitude():

    def __init__(self, threshold):
        self.amplitude_scale = threshold
        self.constant = 10

    def __call__(self, wave):
        amp = self.amplitude_scale * self.constant
        wave = torch.unsqueeze(wave, 0)
        wave = torchaudio.transforms.Vol(gain=amp, gain_type="amplitude")(wave)
        return torch.squeeze(wave, 0)


class WavePower():

    def __init__(self, threshold):
        self.power_scale = threshold
        self.constant = 10

    def __call__(self, wave):
        amp = self.power_scale * self.constant
        wave = torch.unsqueeze(wave, 0)
        wave = torchaudio.transforms.Vol(gain=amp, gain_type="power")(wave)
        return torch.squeeze(wave, 0)


class WaveDB():

    def __init__(self, threshold):
        self.db_scale = threshold
        self.constant = 10

    def __call__(self, wave):
        amp = self.db_scale * self.constant
        wave = torch.unsqueeze(wave, 0)
        wave = torchaudio.transforms.Vol(gain=amp, gain_type="db")(wave)
        return torch.squeeze(wave, 0)


class SpecIdentity():

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, spec):
        return spec


class SpecRandomCrop():

    def __init__(self, threshold):
        self.crop_size = 1 - threshold

    def __call__(self, spec):
        # size = int(spec.shape[1] * self.crop_size)
        size = 1000
        if spec.size(0) < 1000:
            time_pad = (0, 1000-spec.size(0), 0, 0)
            pad = torch.nn.ZeroPad2d(time_pad)
            spec = pad(spec.unsqueeze(0).unsqueeze(0)).squeeze()

        start = random.randint(0, (spec.shape[1] - size))
        return spec[:, start : (start + size)]
        

class SpecFixedCrop():

    def __init__(self, threshold):
        self.crop_size = 1 - threshold

    def __call__(self, spec):
        # size = int(spec.shape[1] * self.crop_size)
        size = 1000
        start = 250
        if spec.size(0) < 1000:
            time_pad = (0, 1000-spec.size(0), 0, 0)
            pad = torch.nn.ZeroPad2d(time_pad)
            spec = pad(spec.unsqueeze(0).unsqueeze(0)).squeeze()
        return spec[:, start : (start + size)]


class SpecGaussianNoise():

    def __init__(self, threshold):
        self.noise_intensity = threshold
        self.constant = 0.2

    def __call__(self, spec):
        noise = self.noise_intensity * self.constant
        return spec + (noise * np.random.normal(size=spec.shape))


class SpecTimeMask():

    def __init__(self, threshold):
        self.mask_size = threshold

    def __call__(self, spec):
        size = int(spec.shape[1] * self.mask_size)
        return torchaudio.transforms.TimeMasking(size)(specgram=spec)


class SpecFreqMask():

    def __init__(self, threshold):
        self.mask_size = threshold

    def __call__(self, spec):
        size = int(spec.shape[0] * self.mask_size)
        return torchaudio.transforms.FrequencyMasking(size)(specgram=spec)


class SpecCheckerNoise():

    def __init__(self, threshold):
        self.mask_size = threshold

    def __call__(self, spec):
        f = SpecFreqMask(self.mask_size)
        t = SpecTimeMask(self.mask_size)
        return f(t(spec))


class SpecFlip():

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, spec):
        return torch.flip(spec, [0])    


class SpecTimeReverse():

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, spec):
        return torch.flip(spec, [1])    


class SpecShuffle():

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, spec):
        # segments = np.array(np.split(spec.T.numpy(), 4, axis=0))
        segments = np.array(np.split(spec.T.numpy()[1:], 3, axis=0))
        np.random.shuffle(segments)

        #used to be reshape(1000,128), but the spec is cropped at top now
        segments =  segments.flatten().reshape(999,128).T
        # segments =  segments.flatten().reshape(1000,128).T
        return torch.from_numpy(segments)
        # return spec

class SpecPermutes():

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, spec):
        segments = np.array(np.split(spec.T.numpy(), 4, axis=0))
        permutes = list(permutations(segments))
        permutes = [numpy.asarray(seg).flatten().reshape(1000,128).T for seg in permutes]
        #returns all permutations of segmented spectrogram

        return torch.from_numpy(np.array(permutes))


# used to access transforms by name
wave_transforms = {
        'wave_identity': WaveIdentity,
        'wave_gaussian_noise': WaveGaussianNoise,
        'wave_amplitude,': WaveAmplitude,
        'wave_db': WaveDB,
        'wave_power': WavePower
}

spec_transforms = {
        'spec_identity': SpecIdentity,
        'spec_gaussian_noise': SpecGaussianNoise,
        'spec_checker_noise': SpecCheckerNoise,
        'spec_flip': SpecFlip,
        'spec_time_reverse': SpecTimeReverse,
        'sepc_shuffle': SpecShuffle
}
