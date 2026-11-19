# %% setup (run from anywhere inside the repo)
from pathlib import Path

import pandas as pd

_here = Path.cwd().resolve()
ROOT = next(p for p in [_here, *_here.parents] if (p / "requirements.txt").exists())
DATA = ROOT / "data" / "housing_messy.csv"

# %% Step 1: load and look


# %% Step 2: write your dtype guesses here first, then check


# %% Step 3: summaries


# %% Step 4: problem hunt (put findings in notes.md)
