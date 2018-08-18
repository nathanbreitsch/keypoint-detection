import os
import imageio
import numpy as np
from toolz import curry
from typing import NamedTuple
from torch.utils.data import Dataset
import torch
from torch import FloatTensor, Size
from toolz import identity
import pandas as pd

class Sample(NamedTuple):
    image: FloatTensor
    keypoints: FloatTensor

class ManifestItem(NamedTuple):
    image_name: str
    keypoints: np.ndarray

class KeypointsDataset(Dataset):

    def __init__(self, manifest, root_dir, transform=identity):
        self.manifest = manifest
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, i):
        item = self.manifest[i]
        image_path = os.path.join(self.root_dir, item.image_name)
        image_ndarray = imageio.imread(image_path)
        image_height, image_width, _ = image_ndarray.shape

        return Sample(
            image = self.transform(image_ndarray_to_tensor(image_ndarray)),
            keypoints = keypoints_ndarray_to_tensor(item.keypoints, image_height, image_width)
        )

# image tensors in this project are always four axis tensors:
# batch, channel, row, column
def image_ndarray_to_tensor(image):
    float_ndarray = image.astype(np.float32)
    raw_tensor = torch.from_numpy(float_ndarray)
    normalized = raw_tensor / 255
    no_alpha = normalized[:, :, :3]
    canonical_torch_axes = no_alpha.permute(2, 0, 1).unsqueeze(0)
    return canonical_torch_axes

def image_tensor_to_ndarray(tensor):
    canonical_numpy_axes = tensor.squeeze(0).permute(1, 2, 0)
    denormalized = 255 * canonical_numpy_axes
    numpy_array = denormalized.numpy()
    int_numpy_array = numpy_array.astype(np.uint8)
    return int_numpy_array

def keypoints_ndarray_to_tensor(keypoints_ndarray, image_height, image_width):
    float_ndarray = keypoints_ndarray.astype(np.float32)
    keypoints_tensor = torch.from_numpy(float_ndarray.copy())
    keypoints_tensor[0::2] *= (2 / image_width)
    keypoints_tensor[1::2] *= (2 / image_height)
    keypoints_tensor -=1
    return keypoints_tensor

def keypoints_tensor_to_ndarray(keypoints, image_height, image_width):
    keypoints_ndarray = keypoints.numpy().copy()
    keypoints_ndarray += 1
    keypoints_ndarray[0::2] *= image_width / 2
    keypoints_ndarray[1::2] *= image_height / 2
    int_ndarray = keypoints_ndarray.astype(np.int32)
    return int_ndarray

def concat_samples(samples):
    return Sample(
        image = torch.cat([s.image for s in samples], dim=0),
        keypoints = torch.cat([s.keypoints.unsqueeze(0) for s in samples], dim=0)
    )

def load_manifest(manifest_path):
    df_raw = pd.read_csv(manifest_path)

    training_manifest = [
        ManifestItem(
            image_name = row[0],
            keypoints = row[1:].values
        ) 
        for _, row in df_raw.iterrows()
    ]
    return training_manifest