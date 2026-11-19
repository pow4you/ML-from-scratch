# 3.3 Evaluation done honestly

**Goal:** know what a score means and what it hides. Accuracy alone has fooled almost everyone at least once.

## Primer

- The **confusion matrix** counts how often true class *i* was predicted as class *j*. Everything
  else is arithmetic on it.
- **Precision** = of everything I flagged, how much was right. **Recall** = of everything real,
  how much did I catch. **F1** is their harmonic mean.
- **Cross-validation** estimates performance by rotating which fold is held out, so every row is
  validated exactly once, and it tells you how much the score *wobbles*.
- With **imbalanced classes** a useless model can score 97% accuracy. The decision threshold (0.5 by default) is a
  business decision, not a law of nature.

Write your code in `metrics.py`. `pytest module-3-applied-ml/ex03-evaluation` checks it.

## Steps

### 🟢 Step 1: the confusion matrix
Implement `confusion_matrix(y_true, y_pred, n_classes)` (rows = true class, columns = predicted).
Fit any classifier on `load_digits` (e.g. `LogisticRegression`), build the matrix on a test split,
plot it as a heat map, and name the most confused pair of digits.

### 🟢 Step 2: precision, recall, F1
Implement `precision_recall_f1(cm)` returning three arrays (one value per class). Compute the
*macro* average (mean over classes) and the *micro* average (pool all counts). When do the two
differ? Compare your numbers with `sklearn.metrics.classification_report`.

### 🟢 Step 3: cross-validation by hand
Implement `kfold_indices(n, k, seed)` that yields `(train_idx, val_idx)` pairs. Every sample must be validated
exactly once. Evaluate a classifier with it. Then draw 20 different random 75/25 splits and plot the spread of
accuracies next to your 5-fold result. What does the spread tell you about trusting a single split?

### 🔴 Step 4: imbalance and the threshold
Generate data with `make_classification(n_samples=5000, weights=[0.97, 0.03], n_informative=4, flip_y=0.01, random_state=0)`.
(a) Show that "always predict negative" has accuracy about 97%, and compare with a logistic regression's accuracy.
(b) Implement the precision–recall curve by sweeping thresholds by hand.
(c) Suppose a missed positive costs 20 and a false alarm costs 1. Choose the threshold that minimises the **total cost on a
validation set**, and report the cost on a separate test set for that threshold vs the default 0.5 vs "always negative".
(d) Repeat with 5 different splits: how stable is the chosen threshold? Write what that means for deploying it.

## Can you explain it?

- Why can a model with 97% accuracy be worthless?
- When would you optimise recall over precision, and when the reverse?
- Why is the threshold chosen on validation data and not on test data?
