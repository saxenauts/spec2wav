import torch
import numpy as np



def reverse(tensor, dim):
    np_arr = np.flip(tensor.numpy(),dim).copy()
    return torch.from_numpy(np_arr)


def upsample(tensor, ratio):
    return old.repeat(1, ratio)
