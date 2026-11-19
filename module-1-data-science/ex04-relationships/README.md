# 1.4 Relationships

**Goal:** understand what a correlation says, what it doesn't say, and how a hidden
third variable can fool you.

## Primer

The Pearson correlation `r` measures how well a *straight line* describes two variables
(−1 to 1). It says nothing about causes, and it is blind to curves.

A **confounder** is a variable that drives both things you are comparing. Example: hot
weather raises ice cream sales and drownings. Ice cream doesn't drown anyone. The standard
way to see through a confounder is to **compare like with like**: group the data by the
confounder and look at the relationship inside each group.

## Steps

### 🟢 Step 1: the correlation table
`df[numeric_columns].corr()`. Which variable correlates most with `price_eur`? Which pairs
of *features* correlate with each other? (This second question matters later: it's why
coefficients get hard to interpret.)

### 🟢 Step 2: groups
Using `groupby("neighborhood")`, compute mean price, mean `sqm` and mean `price_per_sqm`.
Why is mean price a poor way to compare neighborhoods, while price per square metre is better?
Does having a balcony change `price_per_sqm`? By how much, in percent?

### 🟢 Step 3: when is a correlation not a line?
Make a scatter plot of `year_built` vs `price_per_sqm`. Compute `r`. Does `r` match what
you see? Now compute `r` only for flats built after 1950. What changed?

### 🔴 Step 4: the rooms puzzle
`rooms` correlates strongly with `price_eur`. A friend says "add a room, add €X". Test
that claim: bin `sqm` into groups of similar size and compute the `rooms`–`price` relationship
*inside each bin*. What happens to the correlation? What does that say about the claim?

### 🔴 Step 5: write the verdict
In `notes.md`, answer in five sentences or fewer: does the number of rooms drive price in this dataset?
Which variable is the confounder? What experiment would settle it that you can't run
with scraped listings? Then name a second confounded pair you can find in this data.

## Can you explain it?

- Can two variables have `r = 0` and still be perfectly related? Give an example.
- Why does comparing *within groups* control for a confounder?
- What does this mean for reading the coefficients of a model with correlated features?
