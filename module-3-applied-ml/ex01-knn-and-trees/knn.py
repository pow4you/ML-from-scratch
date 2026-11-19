"""Exercise 3.1: k-nearest neighbours with numpy."""

import numpy as np


def knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int) -> np.ndarray:
    """Predict the majority label among the k nearest training points (Euclidean distance).

    Ties in the vote go to the smallest label. No Python loop over test points.
    """
    raise NotImplementedError("your turn")
