# 1.1 Meet your data

**Goal:** build the habit of *looking before touching*. After this exercise you can
inspect any tabular dataset in ten minutes and list what is wrong with it.

The dataset: 600-ish flat listings from Vienna (`data/housing_messy.csv`). Later in the
course you'll predict `price_eur`. It was scraped badly, on purpose.

## Primer

A pandas `DataFrame` is a table where every column has one type. Most data trouble
hides in the types and the encodings: a number stored as text, a yes/no written five ways,
a date in two formats. Your tools are `shape`, `dtypes`, `head`, `info`, `describe`,
`value_counts`, `isna().sum()` and `duplicated().sum()`.

Rule for the whole module: **inspect first, change nothing, write down what you saw.**

## Steps

### 🟢 Step 1: load and look
Load the CSV with `pd.read_csv`. Print the shape, the column names and the first 10 rows.
Write down: what is one row? What would you want to predict?

### 🟢 Step 2: predict the types, then check them
Before running `df.dtypes`, write down the dtype you *expect* for each of the 8 columns.
Then run `df.dtypes` and `df.info()`. Every mismatch is a clue. Which columns are
surprising, and what do you think caused it?

<details><summary>Hint</summary>
Look at the raw CSV text for the surprising column with <code>df["col"].head(20).tolist()</code>.
Mixed content makes pandas fall back to text.
</details>

### 🟢 Step 3: summaries
Use `describe(include="all")`, `value_counts()` on `neighborhood` and `has_balcony`,
`isna().sum()` and `duplicated().sum()`. Look at the min and max of every numeric column
and ask whether a real flat could look like that.

### 🔴 Step 4: the problem hunt
There are **10 kinds** of planted problems in this file. Find as many as you can using
only inspection (no cleaning yet). Create `notes.md` in this folder. For each problem write:

1. what is wrong, in one sentence,
2. the line of pandas that revealed it,
3. what you would do about it and why.

Finding 6 is solid. Finding 8 or more is very good. Don't read `data/make_messy_data.py`
below its marker line.

## Can you explain it?

- Why can a column full of numbers load as text?
- What is the difference between a missing value, an outlier and an impossible value?
  What is a sensible default action for each?
- Why look at the data before cleaning any of it?
