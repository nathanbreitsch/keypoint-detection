from matplotlib import pyplot as plt
from .dataset import image_tensor_to_ndarray, keypoints_tensor_to_ndarray

def plot_image(image_tensor, keypoints_tensor=None, pred_keypoints_tensor=None, ax=None, fig=None):
    if ax is None:
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
        ax.scatter(keypoints_matrix[:, 0], keypoints_matrix[:, 1], marker='.', c='g')

    if pred_keypoints_tensor is not None:
        pred_keypoints_ndarray = keypoints_tensor_to_ndarray(pred_keypoints_tensor, image_height, image_width)
        pred_keypoints_matrix = pred_keypoints_ndarray.reshape((-1, 2))
        ax.scatter(pred_keypoints_matrix[:, 0], pred_keypoints_matrix[:, 1], marker='x', c='r')
        
    return fig, ax