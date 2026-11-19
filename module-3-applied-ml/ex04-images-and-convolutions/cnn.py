"""Exercise 3.4: a small CNN for 8x8 digits."""

import torch
from torch import nn


class DigitsCNN(nn.Module):
    """Conv2d(1->8, 3) -> ReLU -> MaxPool2d(2) -> flatten -> Linear(->10).

    Input (N, 1, 8, 8), output logits (N, 10).
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError("your turn")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("your turn")
