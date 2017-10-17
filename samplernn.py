import torch
from torch.nn import functional as F
from torch.nn import init
from torchqrnn import QRNN
import numpy as np
import utils

class SampleRNN(torch.nn.Module):
    '''
        Builds the complete SampleRNN model on an abstraction. 3 Tier
        Input parameters: top_frame_size,
                          mid_frame_size,
                          top_frame_input_dimensions

    '''

    def __init__(self, top_frame_size, mid_frame_size, top_frame_input_dimensions, \
                    output_dimensions):
        super().__init__()



class TopFrameInput(torch.nn.Module):
    '''
        Takes in the Audio, and the Spectrogram
        Input:  audio: mu-law quantized
                spec:  log scale spectrogram

        -Upsamples the Spectrogram
        -Adds them, and outputs the input to the top frame

    '''

    def __init__(self, ratio_top_input):
        super().__init__()

        self.ratio = ratio_top_input

        #TODO: Hidden Layer Initialization

        self.hidden0_1 = torch.zeros(, dim)
        #self.hidden0_1 = torch.zeros(, dim)
        #self.hidden0_1 = torch.zeros(, dim)
        #self.hidden0_1 = torch.zeros(, dim)

        self.qrnn1 = QRNN(spec_input, spec_hidden, num_layers = 1, window = 2, dropout = 0.4)
        self.qrnn1_b = QRNN(spec_input, spec_hidden, num_layers = 1, window = 2, dropout = 0.4)
        self.qrnn2 = QRNN(spec_input, spec_hidden, num_layers = 1, window = 2, dropout = 0.4)
        self.qrnn2_b = QRNN(spec_input, spec_hidden, num_layers = 1, window = 2, dropout = 0.4)

        #TODO: Weight Initializations for QRNN

        self.conv1d = torch.nn.Conv1d(
                        q_levels,           #TODO: hidden, input dims?
                        q_levels,
                        kernel_size = 2,
                        stride = 1)
        #TODO: Weight Initializations

    def forward(self, audio, spectrogram):

        #TODO: cuda
        spectro_qrnn, hidden_1 = self.qrnn1(spectrogram, hidden0_1)
        spectro_qrnn_b, hidden_1_b = self.qrnn1_b(utils.reverse(spectro_qrnn_1_b, 2), \
                                            hidden0_1_b)
        stack_1 = torch.stack([spectro_qrnn_1, spectro_qrnn_1_b])

        spectro_qrnn, hidden_2 = self.qrnn2(stack_1, hidden0_2)
        spectro_qrnn_b, hidden_2_b = self.qrnn2_b(utils.reverse(stack_1, 2),\
                                            hidden0_2_b)

        stack_2 = torch.stack([spectro_qrnn, spectro_qrnn_b])

        #TODO: Interleaving
        #TODO: spectrogram convolution

        upsampled_spectro = stack_2.repeat(1, 1, ratio)

        #TODO: TODO TODO Verify this convolution method
        audio_input = self.conv1d(audio)
        return audio_input + upsampled_spectro


class TierRNN(torch.nn.Module):
    '''
    Generates RNNs, can be used to generate RNN for different tiers.
    Need to provide the frame_size, input_size, sequence_length, number of rnnns.

    Input: prev rnn cell's hidden state, input samples, upsampled conditioning
    '''
    def __init__(self, frame_size, n_rnn, input_dim, seq_len):
        super.__init__()

        self.frame_size = frame_size
        self.n_rnn = n_rnn

        self.clock_input = torch.nn.Conv1d(
                    in_channels = frame_size,
                    out_channels = input_dim,
                    kernel_size = 1
        )

        #TODO: Initialize weight for clock_input

        self.rnn = torch.nn.GRU(
            input_size = input_dim,
            hidden = input_dim,
            num_layers = n_rnn,
            batch_first = True       #Input, Output provided as (Batch x Sequence x Feature)
        )

        #TODO : Initialize the RNNs

        self.tier_output_upsample = utils.LearnedUpsampling1d(
                in_channels = input_dim,
                out_channels = input_dim,
                kernel_size = frame_size
        )
        #TODO: Check this, and apply Initialization

    def forward(self, prev_samples, upper_tier_conditioning, hidden):

        self.tier_input =
