import torch
from samplernn import SampleRNN
import nn


#TODO : Imports

#TODO : Data Loading

#TODO : Model Building

optimzer = optimizer = gradient_clipping(torch.optim.Adam(model.parameters()))
def loss_function(batch_output, batch_inputs):
    #TODO Does this system work?
    return nn.sequence_nll_loss_bits(batch_output, batch_output)


def train(model, data, optimizer, loss_func):
    for data_batch in dataset:
        optimizer.zero_grad()

        batch_inputs = data_batch[0]
        batch_targets = data_batch[2]  #TODO: Figure out use for reset

        batch_output = model(batch_inputs)

        loss = loss_func(batch_output, batch_targets)
        loss.backward()

        optimizer.step()

def run_training(epochs):
    for epoch in range(epochs):
        train(model, data, optimizer, loss_function)
