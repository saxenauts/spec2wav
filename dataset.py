import utils
import spectro

import torch
from torch.utils.data import (
    Dataset, DataLoader as DataLoaderBase
)

from librosa.core import load
#from natsort import natsorted
from tqdm import *

from os import listdir
from os.path import join



class FolderDataset(Dataset):

    def __init__(self, path_wav, path_spec, hindsight, q_levels, ratio_min=0, ratio_max=1):
        super().__init__()
        self.overlap_len = overlap_len
        self.q_levels = q_levels
        file_names_spec = natsorted(
            [join(path_spec, file_name) for file_name in listdir(path_spec)]
        )
        file_names_wav = natsorted(
            [join(path_wav, file_name) for file_name in listdir(path_spec)]
        )
        self.file_names_spec = file_names_spec[
            int(ratio_min * len(file_names_spec)) : int(ratio_max * len(file_names_spec))
        ]
        self.file_names_wav = file_names_wav[
            int(ratio_min * len(file_names_wav)) : int(ratio_max * len(file_names_wav))
        ]

    def __getitem__(self, index):
        (seq, _) = load(self.file_names_wav[index], sr=None, mono=True)
        wav_tensor = torch.cat([
            torch.LongTensor(self.hindsight) \
                 .fill_(utils.q_zero(self.q_levels)),   #TODO linear_quantize change
            utils.linear_quantize(
                torch.from_numpy(seq), self.q_levels
            )
        ])

        spec_tensor = torch.from_numpy(np.load(self.file_names_spec[index], allow_pickle=False))
        #TODO add hindsight zeros to the spec_tensor

        return wav_tensor, spec_tensor

    def __len__(self):
        return len(self.file_names)


class DataLoader(DataLoaderBase):
    '''
    Takes batches of inputs audio sequences
    Divides them in chunks of bptt_length
    The input sequence is (i-overlap, i+bptt_len -1)
    Target Sequence is (i, i + bptt_len)
    '''


    def __init__(self, dataset, batch_size, seq_len, hindsight, wav_spec_ratio,
                 *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.hindsight = hindsight

    def __iter__(self):
        for batch_wav, batch_spec in super().__iter__():
            (batch_size, n_samples) = batch_wav.size()

            reset = True #TODO: Functionality of reset?

            for seq_begin in range(self.hindsight, n_samples, self.seq_len):
                from_index = seq_begin - self.hindsight
                to_index = seq_begin + self.seq_len
                sequences = batch_wav[:, from_index : to_index]
                input_sequences = sequences[:, : -1]
                target_sequences = sequences[:, self.hindsight :]

                input_specs = batch_spec[:, from_index//wav_spec_ratio: to_index//wav_spec_ratio]

                yield (input_sequences, reset, target_sequences, input_specs)

                reset = False

    def __len__(self):
        raise NotImplementedError()

def main(wav_path, spec_path):
    file_names = [f for f in listdir(wav_path)]

    for file_name in tqdm(file_names):
        wav = spectro.load_wav(join(wav_path, file_name))
        mel_spectrogram = spectro.mel_spectrogram(wav).astype(np.float32)

        #TODO: Saving in time major format?
        np.save(os.path.join(spec_path, file_name), mel_spectrogram.T, allow_pickle=False)
    print ("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--spectro_save_path')
    args = parser.parse_args()
    main(args.data_path, spectro_save_path)
