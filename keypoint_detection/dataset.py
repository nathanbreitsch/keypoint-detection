import os
import imageio
import numpy as np
from toolz import curry
from typing import NamedTuple
from torch.utils.data import Dataset
import torch
from torch import FloatTensor, Size

class Sample(NamedTuple):
    image: FloatTensor
    keypoints: FloatTensor

class ManifestItem(NamedTuple):
    image_name: str
    keypoints: np.ndarray

class KeypointsDataset(Dataset):

    def __init__(self, manifest, root_dir):
        self.manifest = manifest
        self.root_dir = root_dir

    def __len__(self):
        return len(manifest)

    def __getitem__(self, i):
        item = self.manifest[i]
        image_path = os.path.join(self.root_dir, item.image_name)
        image_ndarray = imageio.imread(image_path)

        return Sample(
            image = image_ndarray_to_tensor(image_ndarray),
            keypoints = keypoints_ndarray_to_tensor(item.keypoints)
        )

def image_ndarray_to_tensor(image):
    float_ndarray = image.astype(np.float32)
    raw_tensor = torch.from_numpy(float_ndarray)
    normalized = raw_tensor / 255
    no_alpha = normalized[:, :, :3]
    canonical_torch_axes = no_alpha.permute(2, 0, 1)
    return canonical_torch_axes

def image_tensor_to_ndarray(tensor):
    canonical_numpy_axes = tensor.permute(1, 2, 0)
    denormalized = 255 * canonical_numpy_axes
    numpy_array = denormalized.numpy()
    int_numpy_array = numpy_array.astype(np.uint8)
    return int_numpy_array

def keypoints_ndarray_to_tensor(keypoints_ndarray):
    float_ndarray = keypoints_ndarray.astype(np.float32)
    keypoints_tensor = torch.from_numpy(float_ndarray)
    normalized = (keypoints_tensor - 100) / 50
    return normalized

def keypoints_tensor_to_ndarray(keypoints):
    denormalized = keypoints * 50 + 100
    keypoints_ndarray = denormalized.numpy()
    int_ndarray = keypoints_ndarray.astype(np.int32)
    return int_ndarray


