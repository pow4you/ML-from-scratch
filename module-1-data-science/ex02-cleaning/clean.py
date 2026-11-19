"""Exercise 1.2: write clean(df)."""

from pathlib import Path

import pandas as pd

_here = Path.cwd().resolve()
ROOT = next(p for p in [_here, *_here.parents] if (p / "requirements.txt").exists())
MESSY = ROOT / "data" / "housing_messy.csv"
CLEAN = ROOT / "data" / "housing_clean.csv"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the messy housing table. Do not modify `df`."""
    raise NotImplementedError("your turn")


if __name__ == "__main__":
    out = clean(pd.read_csv(MESSY))
    out.to_csv(CLEAN, index=False)
    print(f"saved {len(out)} rows to {CLEAN}")
