"""Exercise 2.2: logistic neuron, cross-entropy, activations."""

import numpy as np


def sigmoid(z):
    """Numerically stable: must not overflow for z = ±1000."""
    raise NotImplementedError("your turn")


def sigmoid_grad(z):
    raise NotImplementedError("your turn")


def forward(w, b, X):
    """X: (N, d), w: (d,), b: float -> probabilities of shape (N,)."""
    raise NotImplementedError("your turn")


def bce(p, y):
    """Mean binary cross-entropy. Clip p away from exactly 0 and 1."""
    raise NotImplementedError("your turn")


def grads(w, b, X, y):
    """Gradient of the mean BCE: returns (dw of shape (d,), db float)."""
    raise NotImplementedError("your turn")


def train(X, y, lr, steps):
    """Gradient descent from w = 0, b = 0. Return (w, b, losses) with len(losses) == steps."""
    raise NotImplementedError("your turn")


def relu(z):
    raise NotImplementedError("your turn")


def relu_grad(z):
    raise NotImplementedError("your turn")


def leaky_relu(z, alpha=0.01):
    raise NotImplementedError("your turn")


def leaky_relu_grad(z, alpha=0.01):
    raise NotImplementedError("your turn")


def tanh_grad(z):
    raise NotImplementedError("your turn")
