"""Exercise 3.2: an MLP and a training loop in PyTorch."""

import numpy as np
import torch
from torch import nn


class DigitsNet(nn.Module):
    """64 -> hidden -> ReLU -> 10 logits."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        raise NotImplementedError("your turn")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("your turn")


def train(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch_size: int = 32, seed: int = 0) -> list:
    """Train with DataLoader + CrossEntropyLoss + torch.optim.SGD(lr). Return mean loss per epoch."""
    raise NotImplementedError("your turn")


def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of correct predictions; takes and returns plain numpy / Python types."""
    raise NotImplementedError("your turn")
