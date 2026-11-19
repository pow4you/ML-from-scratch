# %% setup
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path.cwd().resolve()
ROOT = next(p for p in [_here, *_here.parents] if (p / "requirements.txt").exists())
CLEAN = ROOT / "data" / "housing_clean.csv"
if not CLEAN.exists():
    CLEAN = ROOT / "data" / "housing_clean_reference.csv"
df = pd.read_csv(CLEAN, parse_dates=["listed_date"])


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# %% Step 1: manual split (write split(df, test_frac, seed) -> train, test)


# %% Step 2: mean and median baselines


# %% Step 3: neighborhood-mean baseline


# %% Step 4: three leaks
