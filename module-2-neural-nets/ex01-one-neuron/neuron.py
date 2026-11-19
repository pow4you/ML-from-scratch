"""Exercise 2.1: one neuron, gradient descent."""

import numpy as np


def predict(w: float, b: float, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("your turn")


def mse(w: float, b: float, x: np.ndarray, y: np.ndarray) -> float:
    raise NotImplementedError("your turn")


def grads(w: float, b: float, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (dL/dw, dL/db) for the mean squared error."""
    raise NotImplementedError("your turn")


def gradient_descent(x, y, lr, steps, w0=0.0, b0=0.0):
    """Return (w, b, losses); losses[i] is the loss before update i (length = steps)."""
    raise NotImplementedError("your turn")
