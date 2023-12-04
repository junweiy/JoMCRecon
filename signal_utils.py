import torch
import torch.fft
import numpy as np
import os

def fft(img):
    # assert img.ndim == 3
    return np.fft.fftshift((np.fft.fft2(img)))


def ifft(k_space_data):
    # assert k_space_data.ndim == 3
    return np.fft.ifft2(np.fft.ifftshift(k_space_data))

def torch_fft(tensor, dim):
    return fftshift(torch.fft.fftn(tensor, dim=dim), dim=dim)

def torch_ifft(tensor, dim):
    return torch.fft.ifftn(ifftshift(tensor, dim), dim=dim)

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return torch.roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return torch.roll(x, shift, dim)


def mkdir(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)