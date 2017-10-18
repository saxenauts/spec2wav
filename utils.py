import torch
import numpy as np
from librosa.core import load
from torch import nn


def reverse(tensor, dim):
    #TODO: What is the input, and what is the return?
    np_arr = np.flip(tensor.numpy(),dim).copy()
    return torch.from_numpy(np_arr)


def upsample(tensor, ratio):
    return old.repeat(1, 1, ratio)

def load(files):
    pass
    '''
    dir_path = /..../
    for wav_path in dir_path:
        wav = load(wav_path, mono = True)
        wav = normalize(wav)
    '''

def normalize(wav):
'''
    max = np.finfo(wav.dtype).max
    min = np.finfo(wav.dtype).min

    wav = (wav - min)/(max - min)
    wav = wav*2. - 1
    return wav
'''
    max = wav.max(0)
    min = wav.min(0)
    #wav = wav.astype('float64', casting = 'safe')
    wav -= min
    wav /= ((max-min)/2.)
    wav -= 1.
    return wav

def mu_law(x, u = 255):
    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    return ((x + 1)/2*u).astype('int16')
    
'''
 x_mu = np.sign(x) * np.log(1 + mu*np.abs(x))/np.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')
'''

class LearnedUpsampling1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()

        self.conv_t = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False
        )

        if bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(out_channels, kernel_size)
            )
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_t.reset_parameters()
        nn.init.constant(self.bias, 0)

    def forward(self, input):
        (batch_size, _, length) = input.size()
        (kernel_size,) = self.conv_t.kernel_size
        bias = self.bias.unsqueeze(0).unsqueeze(2).expand(
            batch_size, self.conv_t.out_channels,
            length, kernel_size
        ).contiguous().view(
            batch_size, self.conv_t.out_channels,
            length * kernel_size
        )
        return self.conv_t(input) + bias
