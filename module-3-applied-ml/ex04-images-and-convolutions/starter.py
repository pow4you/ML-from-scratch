# %% setup
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage  # noqa: F401
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from cnn import DigitsCNN  # noqa: F401
from conv import conv2d, max_pool2d  # noqa: F401

d = load_digits()
images = (d.images / 16.0).astype(np.float32)  # (N, 8, 8)
labels = d.target.astype(np.int64)
img_tr, img_te, y_tr, y_te = train_test_split(images, labels, test_size=0.25, random_state=0, stratify=labels)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

# %% Step 1: convolve a digit with the two Sobel kernels


# %% Step 2: max pooling (and the parameter arithmetic goes in notes.md)


# %% Step 3: the CNN vs your MLP at a similar parameter count


# %% Step 4: augmentation on 200 samples, 5 seeds


# %% Step 5: filters, misclassified digits, saliency map
