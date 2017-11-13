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
                          top_frame_int_dimensions

    '''

    def __init__(self, int_dim, q_levels, ratio_spec2wav, \
                    output_dimensions):
        super().__init__()

    self.int_dim = int_dim
    self.q_levels = q_levels
    self.top_frame_size = 16

    self.top_frame_input = TopFrameInput(ratio_spec2wav)

    self.tiers_rnns = torch.nn.ModuleList([
        TierRNN(
            4, 2, self.int_dim
        ),                                      #TierRNN listed according to tier index
        TierRNN(
            16, 2, self.int_dim
        )
    ])

    self.reset_hidden_states()
    self.mlp = MLP(4, self.int_dim, self.q_levels)

    def forward(self, audio_clip, spectrogram_clip, reset):
        '''
        Input audio_clip has the size of bptt length + top_frame_size
        '''
        if reset: #TODO: Fix reset
            self.reset_hidden_states()

        conditioning = self.top_frame_input(audio_clip, spectrogram_clip)
        input_seq = audio_clip
        for rnn in reversed(self.tiers_rnns):
            from_index = top_frame_size - rnn.frame_size
            to_index = -rnn.frame_size + 1
            prev_samples = input_seq[:, from_index:to_index]
            conditioning = self.run_rnn(rnn, prev_samples, conditioning)

        bottom_frame_size = self.tiers_rnns[0].frame_size
        mlp_input = input_seq[:, self.hindsight - bottom_frame_size : ]

        return self.mlp(mlp_input, upper_tier_conditioning)


    def reset_hidden_states(self):
        self.hidden_states = {rnn: None for rnn in self.tiers_rnns}

    def run_rnn(self, rnn, prev_samples, upper_tier_conditioning):
        (output, new_hidden) = rnn(
                prev_samples, upper_tier_conditioning, self.hidden_states[rnn]
                )
        self.hidden_states[rnn] = new_hidden.detach()
        return output

    @property
    def hindsight(self):
        return self.tiers_rnns[-1].frame_size




class TopFrameInput(torch.nn.Module):
    '''
        Takes in the Audio, and the Spectrogram
        Input:  audio: mu-law quantized
                spec:  log scale spectrogram

        -Upsamples the Spectrogram
        -Adds them, and outputs the input to the top frame

    '''

    def __init__(self, ratio_top_input, spec_dim):
        super().__init__()

        self.ratio = ratio_top_input
        self.spec_input = #TODO
        self.spec_hidden = spec_input

        self.qrnn1 = QRNN(spec_input, num_layers = 1, window = 2, dropout = 0)
        self.qrnn1_b = QRNN(spec_input, num_layers = 1, window = 2, dropout = 0)
        self.qrnn2 = QRNN(spec_input, num_layers = 1, window = 2, dropout = 0)
        self.qrnn2_b = QRNN(spec_input, num_layers = 1, window = 2, dropout = 0)

        init.xavier_uniform(self.qrnn1.weight)
        init.xavier_uniform(self.qrnn1_b.weight)
        init.xavier_uniform(self.qrnn2.weight)
        init.xavier_uniform(self.qrnn2_b.weight)

        self.conv1d = torch.nn.Conv1d(
                        spec_input,
                        int_dim,
                        kernel_size = 1,
                        stride = 1)
        nn.lecun_uniform(self.conv1d.weight)
        init.constant(self.conv1d.bias, 0)

        self.conv2d = torch.nn.Conv1d(
                        1,       #TODO: Find a better way
                        int_dim,
                        kernel_size = 1,
                        stride = 1)
        nn.lecun_uniform(self.conv2d.weight)
        init.constant(self.conv2d.bias, 0)

    def forward(self, audio, spectrogram):

        #TODO: cuda
        batch_size, _ = audio.size()
        self.qrnn_hidden = [torch.zeros(1, batch_size, self.spec_input) for i in range(4)]

        spectro_qrnn, hidden[0] = self.qrnn1(spectrogram, hidden[0])
        spectro_qrnn_b, hidden[1] = self.qrnn1_b(utils.reverse(spectro_qrnn_1_b, 2), \
                                            hidden[1])
        stack_1 = torch.stack([spectro_qrnn_1, spectro_qrnn_1_b])

        spectro_qrnn, hidden[2] = self.qrnn2(stack_1, hidden[2])
        spectro_qrnn_b, hidden[3] = self.qrnn2_b(utils.reverse(stack_1, 2),\
                                            hidden[3])
        #TODO: Interleaving
        stack_2 = torch.stack([spectro_qrnn, spectro_qrnn_b])

        upsampled_spectro = stack_2.repeat(1, 1, ratio)
        upsampled_spectro = self.conv1d(upsampled_spectro)
        audio_input = self.conv2d(audio)
        return audio_input + upsampled_spectro

class TierRNN(torch.nn.Module):
    '''
    Generates RNNs, can be used to generate RNN for different tiers.
    Need to provide the frame_size, input_size, sequence_length, number of rnnns.

    Input: prev rnn cell's hidden state, input samples, upsampled conditioning
    '''
    def __init__(self, frame_size, int_dim, upsample_ratio):
        super.__init__()

        self.frame_size = frame_size
        self.n_rnn = 1

        self.clock_input = torch.nn.Conv1d(
                    in_channels = frame_size,
                    out_channels = int_dim,
                    kernel_size = 1
        )

        init.kaiming_uniform(self.clock_input.weight)
        init.constant(self.clock_input.bias, 0)

        self.rnn = torch.nn.GRU(
            input_size = int_dim,
            hidden = int_dim,
            num_layers = n_rnn,
            batch_first = True       #Input, Output provided as (Batch x Sequence x Feature)
        )

        for i in range(n_rnn): #TODO: Why?
            nn.concat_init(
                getattr(self.rnn, 'weight_ih_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, nn.lecun_uniform]
            )
            init.constant(getattr(self.rnn, 'bias_ih_l{}'.format(i)), 0)

            nn.concat_init(
                getattr(self.rnn, 'weight_hh_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, init.orthogonal]
            )
            init.constant(getattr(self.rnn, 'bias_hh_l{}'.format(i)), 0)

        self.tier_output_upsample = utils.LearnedUpsampling1d(
                in_channels = int_dim,
                out_channels = int_dim,
                kernel_size = upsample_ratio
        )
        init.uniform(
            self.tier_output_upsample.conv_t.weight, -np.sqrt(6 / int_dim), np.sqrt(6 / int_dim)
        )
        init.constant(self.tier_output_upsample.bias, 0)

    def forward(self, prev_samples, upper_tier_conditioning, hidden):
        '''
        Assumed dimensions:
                prev_samples: batch x seq_len_reduced x frame_size
                upper_tier_conditioning: int_dim x seq_len
                hidden: int_dim x seq_len
        '''
        tier_input = self.clock_input(
                    prev_samples.permute(0, 2, 1)
                    ).permute(0, 2, 1)

        if upper_tier_conditioning is not None:
            tier_input += upper_tier_conditioning

        reset = hidden is None

        if hidden is None:
            hidden = zeros(1, int_dim)

        output, hidden = self.rnn(tier_input, hidden)

        output = self.tier_output_upsample(
            output.permute(0, 2, 1)
        ).permute(0, 2, 1)

        return (output, hidden)

class MLP(torch.nn.Module):

    def __init__(self, frame_size, int_dim, q_levels):
        super().__init__()

        self.q_levels = q_levels

        self.embedding = torch.nn.Embedding(
                    self.q_levels,
                    self.q_levels
                    )

        self.input = torch.nn.Conv1d(
            in_channels = q_levels,
            out_channels = int_dim,
            kernel_size = frame_size,
            bias = False
        )
        init.kaiming_uniform(self.input.weight)

        self.hidden = torch.nn.Conv1d(
            in_channels = int_dim,
            out_channels = int_dim,
            kernel_size = 1
        )
        init.kaiming_uniform(self.hidden.weight)
        init.constant(self.hidden.bias, 0)

        self.output = torch.nn.Conv1d(
            in_channels = int_dim,
            out_channels = q_levels,
            kernel_size = 1
        )
        nn.lecun_uniform(self.output.weight)
        init.constant(self.output.bias, 0)

    def forward(self, prev_samples, upper_tier_conditioning):
        '''
        Input: prev_samples : batch_size x sequence_length
               upper_tier_conditioning: batch_size x sequence_length x dim
        Output: log softmax probabilities over q_level values for each sample

        '''
        (batch_size, _) = prev_samples.size()
        prev_samples = self.embedding(prev_samples.contiguous().view(-1)).\
                        view(batch_size, -1, self.q_levels)

        prev_samples = prev_samples.permute(0, 2, 1)
        upper_tier_conditioning = upper_tier_conditioning.permute(0, 2, 1)

        x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
        x = F.relu(self.hidden(x))
        x = self.output(x).permute(0, 2, 1).contiguous()

        # Output  batch_size x sequence_length x q_levels
        return F.log_softmax(x.view(-1, self.q_levels))\
                            .view(batch_size, -1, self.q_levels)

class Generator():

    def __init__(self, params):
        self.params = params

    def __call__(self, model, spectrogram):
        '''
        -Takes in an spectrogram
        -Generates audio of Arbitrary Length
        '''
        #TODO: Volatile
        model.reset_hidden_states()

        bottom_frame_size = model.tiers_rnns[0].frame_size
        seq_len = int(spectrogram.shape[0]*params['ratio_spec2wav'])
        sequences = torch.LongTensor(1, model.hindsight + seq_len)\
                        .fill_(zeros) #TODO: Fill with Q_Levels zeros

        #TODO Wrap all tensors as Variables when feeding to a model.

        upper_tier_conditioning = model.top_frame_input(sequences, spectrogram)

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
