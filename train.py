import torch
from samplernn import SampleRNN
import nn


#TODO : Imports
from dataset import FolderDataset, DataLoader

#TODO : Data Loading
def make_data_loader(params, path_wav, path_spec):
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

#TODO: Results path
#TODO: Model checkpoints Saving and Loading

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

def checkpoints(model):
    torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path))


def main():

    model = SampleRNN(
            input_dim = , #TODO
            q_levels = params['q_levels'],
            ratio_spec2wav = params['ratio_spec2wav'],
            output_dimensions = params['output_dimensions']
    )
    optimizer = gradient_clipping(torch.optim.Adam(model.parameters()))
    #scheduler = ReduceLROnPlateau(optimizer, 'min') TODO: What learning rate schedule?

    data_loader  = make_data_loader(params, path_wav, path_spec)

    test_split = 1 - params['test_frac']
    val_split = test_split - params['val_frac']

    run_training(params['epochs'])
