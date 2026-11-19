"""Exercise 2.4: a two-layer MLP in numpy (tanh hidden, sigmoid output, mean BCE)."""

import numpy as np


def init_params(n_in: int, n_hidden: int, n_out: int, seed: int = 0) -> dict:
    """Return {'W1': (n_in, n_hidden), 'b1': (n_hidden,), 'W2': (n_hidden, n_out), 'b2': (n_out,)}."""
    raise NotImplementedError("your turn")


def forward(params: dict, X: np.ndarray):
    """Return (p, cache) where p has shape (N, n_out)."""
    raise NotImplementedError("your turn")


def bce(p: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy over all entries."""
    raise NotImplementedError("your turn")


def backward(params: dict, cache, y: np.ndarray) -> dict:
    """Return gradients of the mean BCE, same keys and shapes as params."""
    raise NotImplementedError("your turn")


def train(params: dict, X: np.ndarray, y: np.ndarray, lr: float, steps: int) -> list:
    """Full-batch gradient descent. Update params in place; return the losses (len == steps)."""
    raise NotImplementedError("your turn")
