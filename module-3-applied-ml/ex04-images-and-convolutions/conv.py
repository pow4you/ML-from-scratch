"""Exercise 3.4: convolution and pooling in numpy."""

import numpy as np


def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """'Valid' 2-D cross-correlation, stride 1. Output shape (H - kh + 1, W - kw + 1)."""
    raise NotImplementedError("your turn")


def max_pool2d(img: np.ndarray, size: int) -> np.ndarray:
    """Non-overlapping max pooling with stride = size; drop any remainder rows/columns."""
    raise NotImplementedError("your turn")
