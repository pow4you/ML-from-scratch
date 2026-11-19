# %% setup
import time  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from lab_utils import load_split, make_mlp, run  # noqa: F401

# %% Step 1: baseline run and its loss curve


# %% Step 2: learning-rate sweep (log axis!)


# %% Step 3: train/validation curves with partial_fit on 150 samples
# hint: model = make_mlp(); model.partial_fit(X, y, classes=np.arange(10)) for the first call


# %% Step 4: capacity x regularisation heat map


# %% Step 5a: batch sizes against epochs and against time


# %% Step 5b: zero / tiny / large initialisation (use your own numpy MLP from 2.4)


# %% Step 6: random search, 3 seeds, test set touched once (write FINDINGS.md)
