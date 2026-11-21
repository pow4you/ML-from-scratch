# %% setup (run from anywhere inside the repo)
from pathlib import Path

import pandas as pd

_here = Path.cwd().resolve()
ROOT = next(p for p in [_here, *_here.parents] if (p / "requirements.txt").exists())
DATA = ROOT / "data" / "housing_messy.csv"

# %% Step 1: load and look
df = pd.read_csv(DATA)
#print(df)

# print shape
print(df.shape)

# print columns
print(df.columns)

# print first 10 rows
print(df.head(10))

# QA:
# what is one row?
# a row is one flat from the data. it has the same cells but needs cleanup.
#
# What would you want to predict?
# we shouldn't predict FACTS about the apartment, so basically all fields BUT price_eur and id are fundamental, from these two only price_euro makes sense to predict, and that matches up with the exercise description.
# I am confident in my answer of predicting "price_eur".
#

# %% Step 2: write your dtype guesses here first, then check

# The table has the columns:
# 'id', 'neighborhood', 'sqm', 'rooms', 'year_built', 'has_balcony', 'price_eur' and 'listed_date'
#
# The datatypes I would set:
# id: integer
# neighborhood: String (or better, a foreign key to another table of valid places)
# sqm: number
# rooms: integer
# year_built and date_listed: integer (in datetime or unix time)
# price_eur: number

print


# %% Step 3: summaries


# %% Step 4: problem hunt (put findings in notes.md)
