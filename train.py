import torch
from samplernn import SampleRNN
import nn


#TODO : Imports
from dataset import FolderDataset, DataLoader

#TODO : Data Loading

def make_data_loader():
    def data_loader(split_from, split_to, eval):
        dataset = FolderDataset(path_wav, path_spec, params['hindsight'], params['q_levels']
                                    split_from, split_to
                                    )
        return DataLoader(
        dataset,
        batch_size = params['batch_size'],
        seq_len = params['seq_len']
        hindsight = params['hindsight'],
        shuffle=(not eval),
        drop_last=(not eval)
    )
    return data_loader


#TODO : Model Building

#TODO: Model checkpoints Saving and Loading



optimizer = gradient_clipping(torch.optim.Adam(model.parameters()))
def loss_function(batch_output, batch_inputs):
    #TODO Does this system work?
    return nn.sequence_nll_loss_bits(batch_output, batch_output)


def train(model, data, optimizer, loss_func):
    for data_batch in dataset:
        optimizer.zero_grad()

        batch_inputs = data_batch[0]
        batch_targets = data_batch[2]
        batch_spectro = data_batch[3]  #TODO: Figure out use for reset

        batch_output = model(batch_inputs, batch_spectro)

        loss = loss_func(batch_output, batch_targets)
        loss.backward()

        optimizer.step()

def run_training(epochs):
    for epoch in range(epochs):
        train(model, data, optimizer, loss_function)


def main():
