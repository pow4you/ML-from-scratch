# %% setup
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_here = Path.cwd().resolve()
ROOT = next(p for p in [_here, *_here.parents] if (p / "requirements.txt").exists())
CLEAN = ROOT / "data" / "housing_clean.csv"
if not CLEAN.exists():
    CLEAN = ROOT / "data" / "housing_clean_reference.csv"
df = pd.read_csv(CLEAN, parse_dates=["listed_date"])
df["price_per_sqm"] = df["price_eur"] / df["sqm"]

# %% Step 1: correlation table


# %% Step 2: groups


# %% Step 3: year_built vs price_per_sqm


# %% Step 4: rooms inside sqm bins (hint: pd.qcut)


# %% Step 5: verdict goes in notes.md
