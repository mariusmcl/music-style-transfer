import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.io import wavfile

matplotlib.use('agg')


def mu_law(x, mu=255):
    x = np.clip(x, -1, 1)
    x_mu = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return ((x_mu + 1) / 2 * mu).astype('int16')


def inv_mu_law(x, mu=255.0):
    x = np.array(x).astype(np.float32)
    y = 2. * (x - (mu + 1.) / 2.) / (mu + 1.)
    return np.sign(y) * (1. / mu) * ((1. + mu) ** np.abs(y) - 1.)


def cross_entropy_loss(prediction, target):
    # input:  (batch, 256, len)
    # target: (batch, len)

    batch, channel, seq = prediction.size()

    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(-1, 256)  # (batch * seq, 256)
    target = target.view(-1).long()  # (batch * seq)

    cross_entropy = F.cross_entropy(prediction, target, reduction='none')  # (batch * seq)
    return cross_entropy.reshape(batch, seq).mean(dim=1)


class LossMeter(object):
    def __init__(self, name):
        self.name = name
        self.losses = []

    def reset(self):
        self.losses = []

    def add(self, val):
        self.losses.append(val)

    def summarize_epoch(self):
        if self.losses:
            return np.mean(self.losses)
        else:
            return 0

    def sum(self):
        return sum(self.losses)


def wrap_cuda(data, **kwargs):
    if torch.is_tensor(data):
        var = data.cuda(non_blocking=True)
        return var
    else:
        return tuple([wrap_cuda(x, **kwargs) for x in data])


def save_audio(x, path, rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, rate, x)


def save_wav_image(wav, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(15, 5))
    plt.plot(wav)
    plt.savefig(path)
