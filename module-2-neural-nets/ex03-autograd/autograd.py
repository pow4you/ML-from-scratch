"""Exercise 2.3: a tiny scalar autograd engine."""

import math  # noqa: F401  (you will want it for exp and tanh)


class Value:
    def __init__(self, data, _children=(), _op=""):
        raise NotImplementedError("your turn")

    # forward + local derivatives: __add__, __mul__, __neg__, __sub__, __truediv__,
    # __pow__, __radd__, __rmul__, __rsub__, __rtruediv__, exp, tanh, relu

    def backward(self):
        raise NotImplementedError("your turn")


class Neuron:
    """Step 6."""


class Layer:
    """Step 6."""


class MLP:
    """Step 6: MLP(sizes) with parameters() and __call__."""
