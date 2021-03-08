import torch.nn as nn


def _conv1d(in_channel, out_channel, kernel_size, padding):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm1d(out_channel),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )


def _conv1d_3(in_channel, out_channel):
    return _conv1d(in_channel, out_channel, kernel_size=3, padding=1)


def _conv1d_1(in_channel, out_channel):
    return _conv1d(in_channel, out_channel, kernel_size=1, padding=1)
