# 1.5 Splits and baselines

**Goal:** understand why a model is judged on data it has never seen, and why every
model must first beat something stupid.

## Primer

A model that memorises its training data scores perfectly on it and learns nothing.
So we **split**: train on one part, judge on a **test set** that is locked away until the end.

Before building anything, compute a **baseline**: the score of a trivial predictor
(always predict the mean price). If your clever model barely beats it, the model is
not clever. Two metrics: **MAE** (mean absolute error, in euros, easy to read) and
**RMSE** (punishes big misses harder).

**Leakage** is when information from the test set (or from the answer itself) sneaks
into training. It makes scores look great and models useless.

## Steps

### 🟢 Step 1: split by hand
With `np.random.default_rng(seed).permutation`, split the rows 80/20 into train and test.
Check that sizes add up and that no row is in both. Run it twice with the same seed:
same split? Different seed: different split?

### 🟢 Step 2: the baseline
Predict the **training** mean price for every test row. Compute MAE and RMSE on the test set.
Write the numbers down: they are the bar that every later model must clear.
Then try predicting the **median** instead. Which is better in MAE, and why?

### 🟢 Step 3: a slightly smarter baseline
Predict the training mean price **of the row's neighborhood**. Compute MAE and RMSE
again. How much did one categorical column buy you, as a percent of the baseline MAE?

### 🔴 Step 4: build a leak, then explain it
Create a feature set that makes a model look suspiciously good on the test set by
accident-style mistakes. Do it three ways, measuring MAE each time:

1. A blatant leak (hint: look at which columns are *derived from the price*).
2. Computing the neighborhood means on the **whole** dataset before splitting.
3. A third one you can invent or find (think about duplicates, dates, or `id`).

For each, say how the leak happened and how you'd notice it in a real project
when you don't know it's there.

## Can you explain it?

- Why must the baseline's mean come from the training set, not all data?
- Why is MAE easier to explain to a non-technical person than RMSE?
- If your test score is much better than expected, what is the first thing you check?
