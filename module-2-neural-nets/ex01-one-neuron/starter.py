# %% setup
import matplotlib.pyplot as plt
import numpy as np

from neuron import gradient_descent, grads, mse, predict  # noqa: F401

rng = np.random.default_rng(0)

# %% Step 1: data
x = rng.uniform(-3, 3, 50)
y = 3 * x + 2 + rng.normal(0, 0.5, 50)
# plot it, then guess w and b by eye


# %% Step 2: loss at a few points, then a contour plot over (w, b)


# %% Step 3: gradient check with finite differences


# %% Step 4: gradient descent, loss curve, path on the contour


# %% Step 5: largest stable learning rate: experiment vs theory


# %% Step 6a: mini-batch SGD


# %% Step 6b: what unscaled inputs do
