import torch
from samplernn import SampleRNN, Generator
import nn


#TODO : Imports
from dataset import FolderDataset, DataLoader

#TODO: Define Parameters
params = {
'int_dim' : 512,
'hindsight': #TODO,
'q_levels' : 256,
'seq_len' : 512,
'ratio_spec2wav': #TODO

#training parameters

'save_dir': #TODO,
'test_frac': 0.1,
'val_frac': 0.1,
'epochs': 50

}

#TODO : Data Loading
def make_data_loader(params, path_wav, path_spec):
    def data_loader(split_from, split_to, eval):
        dataset = FolderDataset(path_wav, path_spec, params['hindsight'],
                                    params['q_levels'],
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


def spec2wav(generator, model, spectrogram, save_path):

    audio_samples = generator(model, spectrogram)
    wav = utils.mu_law_decoding(audio_samples)
    librosa.output.write_wav(save_path, wav, 16000)

#TODO: Saved model Loading




def loss_function(batch_output, batch_inputs):
    #TODO Does this system work?
    return nn.sequence_nll_loss_bits(batch_output, batch_output)


def train(model, data, optimizer, loss_func):
    for data_batch in dataset:
        optimizer.zero_grad()

        batch_inputs = data_batch[0]
        batch_targets = data_batch[2]
        batch_spectro = data_batch[3]  #TODO:

        batch_output = model(batch_inputs, batch_spectro)

        loss = loss_func(batch_output, batch_targets)
        loss.backward()

        optimizer.step()

def run_training(model, data, optimizer, loss_function, epochs, generator):
    for epoch in range(epochs):
        train(model, data(0, val_split), optimizer, loss_function)

        if epoch%10 == 0:
            save_checkpoints(model, optim, epoch, params['save_dir'])
            #TODO: Setup a random spectrogram and save_path
            spec2wav(generator, model, spectrogram, save_path)



def save_checkpoints(model, optim, epoch, save_dir):
    save_path = os.path.join(save_dir, 'spec2wav_epoch_{}.ckpt'.format(epoch))
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optim},
                 save_path)


def main():

    model = SampleRNN(
            input_dim = , #top_frame_int_dimensions
            q_levels = params['q_levels'],
            ratio_spec2wav = params['ratio_spec2wav']
    )
    #TODO: Grad clipping, optim betas
    optimizer = gradient_clipping(torch.optim.Adam(model.parameters()))

    data_loader  = make_data_loader(params, path_wav, path_spec)
    generator = Generator(params)

    test_split = 1 - params['test_frac']
    val_split = test_split - params['val_frac']
    run_training(model, data_loader, optimizer, loss_function,
                    params['epochs'], generator)
