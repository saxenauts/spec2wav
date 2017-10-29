import torch
from torch import nn



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



def sequence_nll_loss_bits(input, target):
    (_, _, n_classes) = input.size()
    return nn.functional.nll_loss(
        input.view(-1, n_classes), target.view(-1)
    ) * math.log(math.e, 2)
