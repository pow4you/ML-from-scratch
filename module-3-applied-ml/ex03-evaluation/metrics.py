"""Exercise 3.3: evaluation tools, written by hand."""

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """cm[i, j] = number of samples with true class i predicted as class j."""
    raise NotImplementedError("your turn")


def precision_recall_f1(cm: np.ndarray):
    """Return (precision, recall, f1), each an array with one value per class.

    A class that is never predicted (or never occurs) gets 0.0 for the undefined metric.
    """
    raise NotImplementedError("your turn")


def kfold_indices(n: int, k: int, seed: int = 0):
    """Yield (train_idx, val_idx) for k folds over range(n), shuffled with the seed.

    Fold sizes differ by at most one; every index is in exactly one validation fold.
    """
    raise NotImplementedError("your turn")
