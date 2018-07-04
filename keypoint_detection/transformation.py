from .dataset import Sample
from toolz import curry
from torch.nn.functional import grid_sample
import torch
import numpy as np

@curry
def ToGrayscale(image, red_weight=.2126,
                        green_weight=.7152,
                        blue_weight=.0722):
    grayscale = (
        image[:, 0:1, :, :] * red_weight +
        image[:, 1:2, :, :] * green_weight +
        image[:, 2:3, :, :] * blue_weight
    )
    return grayscale


@curry
def Rescale(image, target_height, target_width):
    flowfield_ndarray = np.fromfunction(
        lambda i, j, k: np.where(k == 1, 2 * i / target_height - 1, 2 * j / target_width - 1),
        shape=(target_height, target_width, 2)
    )
    grid = torch.Tensor(flowfield_ndarray).unsqueeze(0)
    return grid_sample(image, grid).data

@curry
def RandomCrop(sample, output_size):
    pass


