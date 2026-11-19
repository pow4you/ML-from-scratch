# %% setup
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

import logistic as lg  # noqa: F401

X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=0)

# %% Step 1: sigmoid and its derivative, plotted


# %% Step 2: forward pass, accuracy at 0.5, decision boundary


# %% Step 3: BCE vs squared error for a true label of 1


# %% Step 4: gradient check, training, boundary at steps 0/10/100/1000


# %% Step 5: gradients through a chain of 10 neurons
