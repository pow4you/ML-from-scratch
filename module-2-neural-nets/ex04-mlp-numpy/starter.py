# %% setup
import matplotlib.pyplot as plt
import numpy as np

import mlp  # noqa: F401
from helpers import numeric_grad, plot_decision_boundary  # noqa: F401

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])

# %% Step 1: parameters and shapes


# %% Step 2: forward pass and untrained loss


# %% Step 3-4: backward and gradient check (use numeric_grad with a closure over params[...])


# %% Step 5: train XOR, loss curve, decision boundary, hidden activations


# %% Step 6: arbitrary depth (write it in deep_mlp.py)


# %% Step 7: digits with softmax + cross-entropy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
Xd = digits.data / 16.0
Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, digits.target, test_size=0.25, random_state=0, stratify=digits.target)
