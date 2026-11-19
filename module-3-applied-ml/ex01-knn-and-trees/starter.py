# %% setup
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from knn import knn_predict  # noqa: F401

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# %% Step 1: k-NN on raw features, k = 1, 3, 5, 15


# %% Step 2: feature spreads, then standardise with TRAIN statistics


# %% Step 3: decision trees and depth


# %% Step 4: the tournament (Pipelines, RepeatedStratifiedKFold)
