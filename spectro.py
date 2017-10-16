import librosa
import librosa.filters
import numpy as np
import io
from scipy import signal
import math


num_mels=80
num_freq=1025
sample_rate=20000
frame_length_ms=50
frame_shift_ms=12.5
rate_preemphasis=0.97
min_level_db=-100
ref_level_db=20
max_iters=200,
griffin_lim_iters=60
power=1.5

def preemphasis(x):
    return signal.lfilter([1, -rate_preemphasis], [1], x)


def inv_preemphasis(x):
    return signal.lfilter([1], [1, -rate_preemphasis], x)

def load_wav(path):
    return librosa.core.load(path, sr = sample_rate)[0]


def spectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return _normalize(S)

def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

#-------------

def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** power))          # Reconstruct phase


def _denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y

def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return n_fft, hop_length, win_length

def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    librosa.output.write_wav(path, wav.astype(np.int16), sample_rate)

wav = load_wav('5.wav')
spectro = spectrogram(wav).astype(np.float32)
print (spectro.shape)
wav_out = inv_spectrogram(spectro)
print (wav_out.shape)
out = "5_result.wav"
save_wav(wav_out, out)
