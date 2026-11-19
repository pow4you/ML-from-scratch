# %% setup
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn  # noqa: F401

from digits_net import DigitsNet, accuracy, train  # noqa: F401

d = load_digits()
X = (d.data / 16.0).astype(np.float32)
y = d.target.astype(np.int64)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# %% Step 1: tensors and autograd


# %% Step 2: the model, parameter count, output shape


# %% Step 3: training loop, then break it (no zero_grad) and swap to Adam


# %% Step 4: match your numpy MLP from 2.4 exactly (float64)


# %% Step 5: regularisation ablation, 300 samples, 5 seeds
