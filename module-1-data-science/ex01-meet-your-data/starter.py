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

df.info()
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
# info shows a lot more data, that is the same data

# it looks like there are mismatched from digitizing old entries from paper with different standards
# i enjoy roll playing as if it was.

# year built shouldn't be a float, a lot of other stuff is string as fallback.

print(df["price_eur"].head(20).tolist())
#i think the stray symbols destroy the data and turn all to strings, plus there is a nan in there as well, weird considering it is al strings
# bonus here is that csvs do not have type enforcement and automatic conversion like excel does.

print(df["has_balcony"].unique())
#ah, the strays of boolean with diffrent writing. merged tables? Who knows?

# %% Step 3: summaries

# Use describe(include="all")
print( df.describe(include="all") )

# value_counts() on neighborhood and has_balcony
print(df["neighborhood"].value_counts())
"""
Leopoldstadt      104
Donaustadt         88
Favoriten          70
Floridsdorf        57
Neubau             57
Josefstadt         45
favoriten          22
DONAUSTADT         17
 Donaustadt        15
leopoldstadt       12
josefstadt         11
floridsdorf        10
FAVORITEN          10
 Favoriten          9
LEOPOLDSTADT        9
neubau              9
donaustadt          9
 Josefstadt         8
NEUBAU              8
 Leopoldstadt       6
FLORIDSDORF         6
 Floridsdorf        5
JOSEFSTADT          5
 FAVORITEN          4
 Neubau             3
 josefstadt         2
 floridsdorf        2
 leopoldstadt       2
 LEOPOLDSTADT       2
 FLORIDSDORF        2
 donaustadt         1
 favoriten          1
 neubau             1
"""

print(df["has_balcony"].value_counts())
"""
N        85
false    70
0        68
yes      65
Yes      63
no       61
No       61
Y        59
1        42
true     38
"""

# isna().sum()
print(df.isna().sum())
"""
id               0
neighborhood     0
sqm             31
rooms            0
year_built      48
has_balcony      0
price_eur       18
listed_date      0
"""

# duplicated().sum()
print(df.duplicated().sum()) # 12
print(df["id"].duplicated().sum())

# min max of all numeric columns
print("sqm:", df["sqm"].min(), df["sqm"].max()) # 2, 9999 (max sounds far too big to be plausible)
print("rooms:", df["rooms"].min(), df["rooms"].max()) # 1,7 (ok)
print("year_built:", df["year_built"].min(), df["year_built"].max()) # 1802, 2124 (2124 is in the future; impossible)





# %% Step 4: problem hunt (put findings in notes.md)
(df["sqm"] / df["rooms"]).describe()