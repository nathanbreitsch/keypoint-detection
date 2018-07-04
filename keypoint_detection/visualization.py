from matplotlib import pyplot as plt
from .dataset import image_tensor_to_ndarray, keypoints_tensor_to_ndarray

def plot_image(image_tensor, keypoints_tensor=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    image_ndarray = image_tensor_to_ndarray(image_tensor)
    image_height, image_width, _ = image_ndarray.shape
    ax.imshow(image_ndarray)
    if keypoints_tensor is not None:
        print(keypoints_tensor.max())
        keypoints_ndarray = keypoints_tensor_to_ndarray(keypoints_tensor, image_height, image_width)
        print(keypoints_ndarray.max())
        keypoints_matrix = keypoints_ndarray.reshape((-1, 2))
        ax.scatter(keypoints_matrix[:, 0], keypoints_matrix[:, 1])

