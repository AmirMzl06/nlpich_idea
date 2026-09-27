"""Additive IID Gaussian noise, with no loss-dependent perturbation."""
import torch
from .model import time_mask


def gaussian_noise(x, lengths, std):
    if std < 0:
        raise ValueError("noise_std must be nonnegative")
    if std == 0:
        return x
    return x + std * torch.randn_like(x) * time_mask(x, lengths)
