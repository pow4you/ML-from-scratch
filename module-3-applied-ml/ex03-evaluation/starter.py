# %% setup
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from metrics import confusion_matrix, kfold_indices, precision_recall_f1  # noqa: F401

# %% Step 1: confusion matrix on digits


# %% Step 2: precision / recall / F1, macro vs micro


# %% Step 3: k-fold by hand vs 20 random splits


# %% Step 4: imbalance, PR curve, cost-based threshold
X_imb, y_imb = make_classification(n_samples=5000, n_features=10, n_informative=4, weights=[0.97, 0.03], flip_y=0.01, random_state=0)
