# 10 kinds of problems

## 1. Missing values
- **What:** `sqm`, `year_built` and `price_eur` have empty cells.
- **Found with:** `df.isna().sum()` (or the Non-Null Count in `df.info()`)
- **Action:** drop rows with no price, since you can't train on them without a target. 

## 2. Wrong column types due to inconsistent content formats
- **What:** `has_balcony`, `year_built` and `listed_date` show up as strings, because their formatting cant be cast to one type automatically.
- **Found with:** `df.dtypes` and `df["price_eur"].head(20).tolist()` 
- **Action:** clean up unparseable symbols from the columns and cast to correct type.

## 3. Outlyers in max on sqm
- **What:** max value in `sqm` is 9999 which is implausible.
- **Found with:** `df["sqm"].max()` 
- **Action:** remove outlyers

## 4. impossible year values
- maximum year is 2124, which is 98 years in the future.
- `df["year_built"].max()`

## 5. duplicated neighborhoods due to casing
`df["neighborhood"].value_counts()`

I think the stray symbols destroy the data and turn all to strings, plus there is a nan in there as well, weird considering it is al strings, bonus here is that csvs do not have type enforcement and automatic conversion like excel does.

## 6. 2 formats in list date
- `list-date` is in formats YYYY-MM-DD and DD.MM.YYYY

## 7. balcony booleans
- `has_balcony` has boolean flag written about 8 different ways
- df["has_balcony"].unique()
- reformat

## 8. duplicated ids and entries
- some ids are duplicated
- `df.duplicated().sum()`, `df["id"].duplicated().sum()`


## 9. Untrimmed strings in neighborhood




# Can you explain it?

## Why can a column full of numbers load as text?
Because pandas defaults to it for formating mismatches and non-numerical symbols.

## What is the difference between a missing value, an outlier and an impossible value? What is a sensible default action for each?


## Why look at the data before cleaning any of it?

