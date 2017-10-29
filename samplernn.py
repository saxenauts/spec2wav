import torch
from torch.nn import functional as F
from torch.nn import init
#from torchqrnn import QRNN
import numpy as np
#import utils

class SampleRNN(torch.nn.Module):
    '''
        Builds the complete SampleRNN model on an abstraction. 3 Tier
        Input parameters: top_frame_size,
                          mid_frame_size,
                          top_frame_input_dimensions

    '''

    def __init__(self, input_dim, q_levels, n_rnn, ratio_spec2wav, \
                    output_dimensions):
        super().__init__()

    self.input_dim = input_dim
    self.q_levels = q_levels
    self.top_frame_size = 16

    self.top_frame_input = TopFrameInput(ratio_spec2wav)

    self.tiers_rnns = torch.nn.ModuleList([
        TierRNN(
            4, 2, self.input_dim
        ),                                      #TierRNN listed according to tier index
        TierRNN(
            16, 2, self.input_dim
        )
    ])

    self.mlp = MLP(4, self.input_dim, self.q_levels)

    def forward(self, audio_clip, spectrogram_clip):
        '''
        Input audio_clip has the size of bptt length + top_frame_size
        '''
        conditioning = self.top_frame_input(audio_clip, spectrogram_clip)
        input_seq = audio_clip
        for rnn in reversed(self.tiers_rnns):
            from_index = top_frame_size - rnn.frame_size
            to_index = -rnn.frame_size + 1

            prev_samples = input_seq[:, from_index:to_index]
            conditioning = run_rnn(rnn, conditioning, prev_samples) #TODO run_rnn function

        mlp_input = input_seq[:, top_frame_size - bottom_frame_size] #TODO

        return self.mlp(mlp_input)


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

        #self.hidden0_1 = torch.zeros(, dim)
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
    def __init__(self, frame_size, n_rnn, input_dim):
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
        '''
        Assumed dimensions:
                prev_samples: input_dim x seq_len
                upper_tier_conditioning: input_dim x seq_len
                hidden: input_dim x seq_len
        '''
        tier_input = clock_input(
                    prev_samples.permute(0, 2, 1)
                    ).permute(0, 2, 1)

        if upper_tier_conditioning is not None:
            tier_input += upper_tier_conditioning

        if hidden is None:
            pass
            #TODO: Initialization of hidden layer
            #      Depends on the number of layer

        output, hidden = self.rnn(tier_input, hidden)

        output = self.tier_output_upsample(
            output.permute(0, 2, 1)
        ).permute(0, 2, 1)

        return (output, hidden)

class MLP(torch.nn.Module):

    def __init__(self, frame_size, input_dim, q_levels):
        super().__init__()

        self.q_levels = q_levels

        self.embedding = torch.nn.Embedding(
                    self.q_levels,
                    self.q_levels
                    )

        self.input = torch.nn.Conv1d(
            in_channels = q_levels,
            out_channels = input_dim,
            kernel_size = frame_size,
            bias = False
        )
        #TODO: Weight Initialization

        self.hidden = torch.nn.Conv1d(
            in_channels = input_dim,
            out_channels = input_dim,
            kernel_size = 1
        )
        #TODO: Weight Initialization

        self.output = torch.nn.Conv1d(
            in_channels = input_dim,
            out_channels = q_levels,
            kernel_size = 1
        )
        #TODO: Weight Initialization

    def forward(self, prev_samples, upper_tier_conditioning):

        #TODO: Batch size
        #TODO: What is prev_samples? dim?

        #TODO
        '''
        WHY SO MANY PERMUTES? WHAT IS BEING FED IN, AND WHAT IS COMING OUT?
        '''
        prev_samples = self.embedding(prev_samples).\
                        view(batch_size, -1, self.q_levels)

        prev_samples = prev_samples.permute(0, 2, 1)
        upper_tier_conditioning = upper_tier_conditioning.permute(0, 2, 1)

        x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
        x = F.relu(self.hidden(x))
        x = self.output(x).permute(0, 2, 1).contiguous()

        # Output  batch_size x sequence_length x q_levels

        #TODO: Check for a different return based on personal code
        return F.log_softmax(x.view(-1, self.q_levels))\
                            .view(batch_size, -1, self.q_levels)

class Generator():

    def __init__(self, model, cuda=False):
        self.cuda = cuda

    def __call__(self, n_seqs, seq_len, spectrogram):
        '''
        -Takes in an spectrogram
        -Generates audio of Arbitrary Length
        '''
        runner.reset_hidden_states #TODO reset hidden states

        bottom_frame_size = model.tiers_rnns[0].frame_size
        sequences = torch.LongTensor(n_seqs, model.hindsight + seq_len)\
                        .fill_(zeros) #TODO: Fill with Q_Levels zeros

        upper_tier_conditioning = model.conv1d(spectrogram) #TODO

        #TODO Implement model hindsight
        for i in range(model.hindsight, model.hindsight + seq_len):
            for rnn in reversed(model.tiers_rnns):
                if i % rnn.frame_size != 0:
                    continue
                prev_samples = sequences[:, i - rnn.frame_size: i]
                #TODO
                upper_tier_conditioning = \
                    run_rnn(rnn, prev_samples, upper_tier_conditioning)

            prev_samples = sequences[:, i - bottom_frame_size : i]
            sample_dist = self.model.mlp(
                prev_samples, upper_tier_conditioning
            )

            sequences = [:, i] = sample_dist.multinomial(1).squeeze(1) #TODO

        return sequences[:, model.hindsight : ]
