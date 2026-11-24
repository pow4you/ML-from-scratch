"""Exercise 1.2: write clean(df)."""

from pathlib import Path

import pandas as pd

_here = Path.cwd().resolve()
ROOT = next(p for p in [_here, *_here.parents] if (p / "requirements.txt").exists())
MESSY = ROOT / "data" / "housing_messy.csv"
CLEAN = ROOT / "data" / "housing_clean.csv"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the messy housing table. Do not modify `df`."""
    df = df.copy()

    # remove empty and duplicates

    # step 1: make neighborhood consistent
    df["neighborhood"] = df["neighborhood"].str.strip().str.title()

    # step 2: turn balcony into a bool
    df["has_balcony"] = df["has_balcony"].str.strip().str.lower().map({
        "1": True, "yes": True, "y": True, "true": True,
        "0": False, "no": False, "n": False, "false": False
    })

    # step 3:
    df["price_eur"] = (
        df["price_eur"]
            .str.extract(r"(\d{5,6}|\d{2,3}\.\d{3})", expand=False)
            .str.replace(".", "", regex=False)
            .astype(float)
    )
    #df["listed_date"] = pd.to_datetime(df["listed_date"], format="mixed")
    iso = pd.to_datetime(df["listed_date"], format="%Y-%m-%d", errors="coerce")
    eu = pd.to_datetime(df["listed_date"], format="%d.%m.%Y", errors="coerce")
    df["listed_date"] = iso.fillna(eu)

    # step 4: freshholds
    df.loc[(df["sqm"] < 5) | (df["sqm"] > 250), "sqm"] = pd.NA #above 5sqm is big enough for one bed and below 2599 seems reasonable.
    df.loc[(df["year_built"] < 1850) | (df["year_built"] > 2026), "year_built"] = pd.NA # only ones that were built in the future
    df = df.drop_duplicates()

    # step 5: missing values
    df.fillna({
        "sqm": df["sqm"].median(),
        "year_built": df["year_built"].median()
    }, inplace=True)

    df = df.dropna(subset=['sqm', 'price_eur', 'year_built']) # results in 500 lines with no missing fields


    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    out = clean(pd.read_csv(MESSY))
    out.to_csv(CLEAN, index=False)
    print(f"saved {len(out)} rows to {CLEAN}")
