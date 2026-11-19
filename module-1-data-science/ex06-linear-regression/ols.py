"""Exercise 1.6: ordinary least squares with numpy only."""

import numpy as np


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit y ≈ b0 + X @ w. X has shape (n, k) WITHOUT an intercept column.

    Return beta of shape (k + 1,) with beta[0] = intercept.
    """
    raise NotImplementedError("your turn")


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return predictions of shape (n,)."""
    raise NotImplementedError("your turn")
