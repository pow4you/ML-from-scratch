# 1.3 Seeing the data

**Goal:** learn to ask a question of the data and answer it with a picture, instead of
making plots until something looks interesting.

Data: `data/housing_clean.csv` (yours from 1.2), or `data/housing_clean_reference.csv`.

## Primer

A plot answers one question. Before each plot, write the question in a comment.

- **Distribution of one variable:** histogram (try different bin counts; try a log axis for money).
- **Two numeric variables:** scatter plot (with transparency when there are many points).
- **Numeric by category:** box plot or strip plot.

Always label axes with units. Skewed money data usually looks better and is easier to model on a log scale.

## Steps

### 🟢 Step 1: one variable at a time
Histograms of `price_eur`, `sqm` and `year_built`. Try 10, 30 and 100 bins on each. What
changes? Which of the three is skewed, and in which direction? Redo `price_eur` with a log-scaled x-axis.

### 🟢 Step 2: two variables
Scatter plot `sqm` vs `price_eur`. Set `alpha=0.4`. Describe the shape in one sentence:
linear? fan-shaped? Where is the spread biggest and why might that be?

### 🟢 Step 3: numbers by category
Box plot of `price_eur` per neighborhood, ordered by median price. Then the same for
`price_eur / sqm`. Which of the two plots tells you more about the neighborhoods, and why?

### 🔴 Step 4: tell one story
Make **one figure** (a 2×2 grid at most) that makes a single non-obvious point about this
market. It needs a title that states the point (not "Price vs sqm"), labelled axes with units,
no chart junk, and a caption in `notes.md` that explains what a stranger should see.
Save it as `figures/story.png`.

## Can you explain it?

- Why does the histogram of price look different on a log axis, and what does that say about how prices behave?
- What can a box plot hide that a strip plot would show?
- What makes a plot misleading? Find one way you could have misled with your own figure.
