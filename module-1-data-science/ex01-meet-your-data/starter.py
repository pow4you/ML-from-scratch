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
# has_balcony: boolean
# year_built and date_listed: integer (in datetime or unix time)
# price_eur: number
#
# from the output of "print(df.columns)" i suspect its all strings tho
#

print(df.dtypes)

# output:
"""
id                int64
neighborhood        str
sqm             float64
rooms             int64
year_built      float64
has_balcony         str
price_eur           str
"""
# suprising...

print(df.info())
"""
id                int64
neighborhood        str
sqm             float64
rooms             int64
year_built      float64
has_balcony         str
price_eur           str
listed_date         str
dtype: object
<class 'pandas.DataFrame'>
RangeIndex: 612 entries, 0 to 611
Data columns (total 8 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   id            612 non-null    int64  
 1   neighborhood  612 non-null    str    
 2   sqm           581 non-null    float64
 3   rooms         612 non-null    int64  
 4   year_built    564 non-null    float64
 5   has_balcony   612 non-null    str    
 6   price_eur     594 non-null    str    
 7   listed_date   612 non-null    str    
dtypes: float64(2), int64(2), str(4)
memory usage: 38.4 KB
None
"""
# info shows a lot more data

# it looks like there are mismatched from digitising old entries from paper with diffrent standards

# %% Step 3: summaries


# %% Step 4: problem hunt (put findings in notes.md)
