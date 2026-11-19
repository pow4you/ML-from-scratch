# 1.6 Linear regression by hand

**Goal:** fit a real model with nothing but numpy, and understand exactly what "fitting" means.
This is the bridge to Module 2: the loss you minimise here in one step is the loss
you will minimise by gradient descent there.

## Primer

Linear regression predicts `ŷ = b₀ + b₁x₁ + … + b_k x_k`. "Fitting" means choosing the
`b`s that minimise the sum of squared errors. For one feature there is a formula (slope =
cov(x, y) / var(x)). For many, stack the features into a matrix **X** (with a column of
ones for the intercept) and solve the **normal equation**:

```
(XᵀX) β = Xᵀy
```

Use `np.linalg.solve` or `np.linalg.lstsq`, never an explicit matrix inverse.
Interpretation: each coefficient is the change in prediction for one unit of that feature, holding the others fixed.

## Steps

Write your functions in `ols.py`. `pytest module-1-data-science/ex06-linear-regression`
checks them.

### 🟢 Step 1: one feature, by formula
Predict `price_eur` from `sqm` using slope = cov/var and intercept = ȳ − slope·x̄.
Plot the scatter with your line. What does the slope mean in euros?

### 🟢 Step 2: the same fit, as linear algebra
Build X = [1, sqm] and solve the normal equation. You should get the same two numbers as in step 1
(to many decimals). Implement `fit_ols(X, y)` and `predict(X, beta)` in `ols.py`
(`X` has *no* intercept column; `beta[0]` is the intercept).

### 🟢 Step 3: judge it honestly
Split with your function from 1.5, fit on train, evaluate on test. Report MAE and **R²**
(1 − SSE/SST). Compare the MAE to your baselines from 1.5. How much better than
"predict the neighborhood mean" is this?

### 🔴 Step 4: many features
Add `year_built`, `has_balcony` (as 0/1) and `neighborhood` (one-hot, **drop one category**:
work out why). Fit on train, test on test. Compare coefficients and predictions
with `sklearn.linear_model.LinearRegression` to at least 6 decimals.

### 🔴 Step 5: what's left over
Plot residuals against fitted values, and also the average residual per fitted-value
quintile. There is structure left (look for curvature, not just spread). Explain it
in terms of how prices actually work: a flat's price is roughly *size × price per
square metre*, and the second factor is multiplied by neighborhood and balcony effects.
Redesign the model with logs so those effects become additive. Decide **which variables**
to take the log of (try the target alone first and see whether it helps), and compare
MAE **in euros**, not in log units, before and after.
Then standardise the features (subtract mean, divide by std) and show the predictions don't
change but the coefficients become comparable. Why?

## Can you explain it?

- Why does the normal equation give the minimum, and what would break if two features were perfect copies?
- Why drop one one-hot column?
- Why is the loss for linear regression a bowl with a single bottom? (You'll need this in Module 2.)
