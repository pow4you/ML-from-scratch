"""Generate data/housing_messy.csv, a deliberately dirty dataset of Vienna flat listings.

Run from the repo root:  python data/make_messy_data.py

The data is synthetic and seeded, so everyone gets the same file.
Module 1, exercise 1 asks you to find the planted problems yourself:
do NOT read the section below the 'messing it up' marker before you try.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "housing_messy.csv"
SEED = 42
N = 600

rng = np.random.default_rng(SEED)

# --- the clean world ---------------------------------------------------------
HOODS = {
    "Leopoldstadt": 1.00,
    "Neubau": 1.35,
    "Favoriten": 0.80,
    "Donaustadt": 0.85,
    "Josefstadt": 1.40,
    "Floridsdorf": 0.78,
}
names = list(HOODS)
hood = rng.choice(names, N, p=[0.20, 0.15, 0.20, 0.20, 0.10, 0.15])
sqm = np.clip(rng.normal(70, 25, N), 20, 200).round(1)
# rooms follow size; price depends on size, not on rooms (useful in Module 1, ex 4)
rooms = np.clip(np.round(sqm / 28 + rng.normal(0, 0.6, N)), 1, 7).astype(int)
year_built = rng.integers(1890, 2024, N)
balcony = rng.random(N) < 0.45

factor = np.array([HOODS[h] for h in hood])
price_per_sqm = 4200 * factor * (1 + 0.12 * balcony) * (1 - 0.0006 * (2024 - year_built))
price_per_sqm *= np.exp(rng.normal(0, 0.08, N))  # multiplicative noise
price = (sqm * price_per_sqm / 1000).round() * 1000

listed = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, N), unit="D")

df = pd.DataFrame(
    {
        "id": np.arange(1000, 1000 + N),
        "neighborhood": hood,
        "sqm": sqm,
        "rooms": rooms,
        "year_built": year_built.astype(float),
        "has_balcony": balcony,
        "price_eur": price,
        "listed_date": listed,
    }
)

# --- messing it up ------------------------------------------------------------
# (stop reading here if you want to find the problems yourself)
df["neighborhood"] = df["neighborhood"].astype(object)
for i in rng.choice(N, 90, replace=False):
    df.at[i, "neighborhood"] = df.at[i, "neighborhood"].lower()
for i in rng.choice(N, 60, replace=False):
    df.at[i, "neighborhood"] = df.at[i, "neighborhood"].upper()
for i in rng.choice(N, 60, replace=False):
    df.at[i, "neighborhood"] = " " + df.at[i, "neighborhood"] + " "

yes = ["yes", "Yes", "Y", "1", "true"]
no = ["no", "No", "N", "0", "false"]
df["has_balcony"] = [rng.choice(yes) if b else rng.choice(no) for b in df["has_balcony"]]

df["price_eur"] = df["price_eur"].astype(object)
for i in rng.choice(N, 180, replace=False):
    v = int(df.at[i, "price_eur"])
    df.at[i, "price_eur"] = "€ " + f"{v:,}".replace(",", ".")

df["listed_date"] = df["listed_date"].astype(object)
for i in rng.choice(N, 200, replace=False):
    df.at[i, "listed_date"] = df.at[i, "listed_date"].strftime("%d.%m.%Y")
for i in df.index:
    if not isinstance(df.at[i, "listed_date"], str):
        df.at[i, "listed_date"] = df.at[i, "listed_date"].strftime("%Y-%m-%d")

df.loc[rng.choice(N, 48, replace=False), "year_built"] = np.nan
df.loc[rng.choice(N, 30, replace=False), "sqm"] = np.nan
df.loc[rng.choice(N, 18, replace=False), "price_eur"] = np.nan

outlier_rows = rng.choice(N, 8, replace=False)
df.loc[outlier_rows[:3], "sqm"] = 9999.0
df.loc[outlier_rows[3:5], "sqm"] = df.loc[outlier_rows[3:5], "sqm"] * 10  # decimal typo
df.loc[outlier_rows[5:], "sqm"] = 2.0
bad_year_rows = rng.choice(N, 5, replace=False)
df.loc[bad_year_rows[:2], "year_built"] = 1802.0
df.loc[bad_year_rows[2:], "year_built"] = 2124.0

dupes = df.sample(12, random_state=SEED)
df = pd.concat([df, dupes], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

df.to_csv(OUT, index=False)
print(f"wrote {OUT} ({len(df)} rows)")
