# 1.2 Cleaning

**Goal:** turn the messy file into a table you can trust, with every decision written
down and defended. The output of this exercise (`clean()`) is reused by the rest of
the course.

## Primer

Cleaning is a sequence of *decisions*, not a sequence of commands. For each problem you choose:
**fix it** (the true value is recoverable), **flag it** (set to missing), **impute it**
(fill with a defensible guess) or **drop it** (the row is useless).

Two rules: never edit the original (work on a copy), and never impute the thing you will
later predict (`price_eur`): a row with no price can't teach a price model anything.

Useful tools: `str.strip`, `str.title`, `str.lower`, `isin`, `map`, `str.replace`,
`pd.to_numeric`, `pd.to_datetime`, `drop_duplicates`, `where`, `fillna`, `groupby(...).transform`.

## Steps

Put your code in `clean.py` as one function `clean(df) -> DataFrame`.
`pytest module-1-data-science/ex02-cleaning` tells you when the basics hold.

### 🟢 Step 1: text columns
Make `neighborhood` consistent (case and whitespace). Verify with `value_counts()` that
exactly six names remain.

### 🟢 Step 2: yes/no column
`has_balcony` is written five different ways per answer. Turn it into a real boolean.
Make sure you handle *every* spelling, and think about what you'd do with a spelling you've never seen.

### 🟢 Step 3: numbers and dates
Make `price_eur` numeric and `listed_date` a real datetime.
Careful: the dot means *thousands separator* in some prices and *decimal point* in others.
Check your parsed prices against the raw ones for both styles. The two date formats
are `2024-03-15` and `15.03.2024`.

<details><summary>Hint</summary>
Handle the two price styles separately instead of stripping all non-digits. For dates,
parse with an explicit <code>format=</code> twice and combine the results.
</details>

### 🔴 Step 4: duplicates and impossible values
Drop exact duplicates. Then decide what "impossible" means for `sqm` and `year_built`.
Look at the distributions to choose thresholds, set impossible values to missing, and write
a sentence per threshold explaining your choice. (The tests allow a fairly wide range:
the point is that you can defend yours.)

### 🔴 Step 5: missing values
Decide per column: drop, impute, or leave? Implement it. Constraints: the result must
have no missing values, must keep at least 500 rows, and must not invent prices.
Compare two imputation strategies for `sqm` (overall median vs median per neighborhood)
and explain which you prefer.

When `clean.py` is run as a script it saves `data/housing_clean.csv`.
(If you get really stuck, `data/housing_clean_reference.csv` exists so you can keep
going, but try to finish this one yourself first.)

## Can you explain it?

- Why must `clean()` not modify its input?
- When is imputing a mistake? (Name two situations.)
- What would have gone wrong if you had stripped every non-digit from the prices?
