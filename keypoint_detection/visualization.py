from matplotlib import pyplot as plt
from .dataset import image_tensor_to_ndarray, keypoints_tensor_to_ndarray

def plot_image(image_tensor, keypoints_tensor=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    image_ndarray = image_tensor_to_ndarray(image_tensor)
    image_height, image_width, n_channels = image_ndarray.shape

    if n_channels == 1:
        ax.imshow(image_ndarray.squeeze(2), cmap='gray')
    else:
        ax.imshow(image_ndarray)

    if keypoints_tensor is not None:
        keypoints_ndarray = keypoints_tensor_to_ndarray(keypoints_tensor, image_height, image_width)
        keypoints_matrix = keypoints_ndarray.reshape((-1, 2))
        ax.scatter(keypoints_matrix[:, 0], keypoints_matrix[:, 1])

