import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

#TODO : Imports
from dataset import FolderDataset, DataLoader
from samplernn import SampleRNN, Generator
import nn

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
'epochs': 50,
'resume': None
}

def Variable_wrap(input):
     if torch.is_tensor(input):
         input = Variable(input)
         if CUDA:
             input = input.cuda()
         return input

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


def save_checkpoints(model, optim, epoch, save_dir):
    save_path = os.path.join(save_dir, 'spec2wav_epoch_{}.ckpt'.format(epoch))
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state': optim.state_dict()},
                 save_path)

def load_last_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    model.load_state_dict(checkpoint['state_dict'])
    return epoch, model, optimizer

#TODO: Saved model Loading
#TODO: Global CUDA flag


def loss_function(batch_output, batch_inputs):
    #TODO How Does this system work?
    return nn.sequence_nll_loss_bits(batch_output, batch_output)


def train(model, dataset, optimizer, loss_func, epoch):
    loss_plot = []
    total_loss = 0
    for i, data_batch in enumerate(dataset):
        optimizer.zero_grad()

        batch_size = data_batch.shape[0]

        batch_inputs = list(map(Variable_wrap, data_batch[0]))
        batch_targets = list(map(Variable_wrap, data_batch[2]))
        batch_spectro = list(map(Variable_wrap, data_batch[3]))

        batch_output = model(batch_inputs, batch_spectro)

        loss = loss_func(batch_output, batch_targets)
        loss.backward()

        optimizer.step()

        total_loss += loss/batch_size
        if i%100:
            loss_plot.append(total_loss/100)
            total_loss = 0

    return loss_plot


def evaluate(model, dataset, loss_func, epoch):
    model.eval()

    loss_plot = []
    total_loss = 0
    for (i, data) in enumerate(dataset):

        batch_size = data.shape[0]

        batch_inputs = list(map(Variable_wrap, data_batch[0]))
        batch_targets = list(map(Variable_wrap, data_batch[2]))
        batch_spectro = list(map(Variable_wrap, data_batch[3]))

        batch_output = model(batch_inputs, batch_spectro)
        loss = loss_func(batch_output, batch_targets)
        total_loss += loss/batch_size
        if i%100 == 0:
            loss_plot.append(total_loss/100)
            total_loss = 0

    model.train()
    return loss_plot


def run_training(model, data, optimizer, loss_function, epochs, generator):
    t_loss = []
    e_loss = []
    for epoch in range(epochs):
        train_loss = train(model, data(0, val_split), optimizer, loss_function, epoch)
        eval_loss = evaluate(model, data(val_split, 1), loss_function, epoch)

        t_plot = np.asarray(t_loss.append(train_loss))
        e_plot = np.asarray(e_loss.append(eval_loss))
        t_plot = np.reshape(-1, t_plot.shape[-1])
        e_plot = np.reshape(-1, e_plot.shape[-1])

        iters = np.arange(0, t_plot.shape[0]*100, 100)
        plt.plot(t_plot, iters, 'r', e_plot, 'b', linewidth=2.0)
        plt.save_fig(params['plot_save_path'])
        plt.clf()

        if epoch%10 == 0:
            save_checkpoints(model, optim, epoch, params['save_dir'])
            #TODO: Setup a random spectrogram and save_path
            spec2wav(generator, model, spectrogram, save_path)


def main():

    model = SampleRNN(
            input_dim = , #top_frame_int_dimensions
            q_levels = params['q_levels'],
            ratio_spec2wav = params['ratio_spec2wav']
    )
    if CUDA:
        model = model.cuda()

    #TODO: Grad clipping, optim betas
    optimizer = gradient_clipping(torch.optim.Adam(model.parameters()))

    data_loader  = make_data_loader(params, path_wav, path_spec)
    generator = Generator(params)

    test_split = 1 - params['test_frac']
    val_split = test_split - params['val_frac']
    if params['resume']:
        epoch, model, optimizer = load_last_checkpoint(params['resume'])
    run_training(model, data_loader, optimizer, loss_function,
                    params['epochs'], generator)
