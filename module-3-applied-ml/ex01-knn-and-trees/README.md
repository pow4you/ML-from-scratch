# 3.1 k-NN and trees

**Goal:** meet two completely different kinds of model, learn why feature scaling matters for one and
not the other, and learn to compare models fairly instead of by gut feeling.

## Primer

- **k-nearest neighbours** keeps the training data and predicts by majority vote among the `k`
  closest training points. "Closest" means distance, so the *units* of the features decide who
  counts.
- A **decision tree** splits the data again and again with questions like "proline > 755?". It
  doesn't care about scale, can fit anything (and so overfits), and is easy to read.
- A **random forest** averages many trees trained on resampled data; it trades readability for stability.

Write your k-NN in `knn.py`. `pytest module-3-applied-ml/ex01-knn-and-trees` checks it.

## Steps

### 🟢 Step 1: k-NN by hand
Implement `knn_predict(X_train, y_train, X_test, k)` using numpy broadcasting (no loops over
test points) to get all pairwise distances. Load `sklearn.datasets.load_wine`, split 70/30 with
`stratify=y`, and measure accuracy for `k = 1, 3, 5, 15` on the raw features.

<details><summary>Hint</summary>
<code>np.argsort</code> on each row of the distance matrix gives neighbours in order. For the
vote, <code>np.bincount</code> on the neighbours' labels, then <code>argmax</code>.
</details>

### 🟢 Step 2: scaling
Print the standard deviation of each wine feature. One of them is hundreds of times bigger than
most. Standardise with the **training** mean and std and rerun step 1. What happened to the accuracy,
and why? Which feature was dominating the distance before?

### 🟢 Step 3: trees
Fit `DecisionTreeClassifier(max_depth=d)` for `d = 1 … 10`. Plot train and test accuracy against depth.
Print the tree with `sklearn.tree.export_text` for a depth that generalises well and read it aloud
as rules. Check `feature_importances_`. Then fit the same tree on scaled features: do the predictions change? Why not?

### 🔴 Step 4: a fair tournament
Compare **k-NN, a decision tree, a random forest and logistic regression** on three datasets:
wine, breast cancer, digits. Rules: identical folds for every model (repeated stratified k-fold, e.g. 5×2),
scaling inside a `Pipeline` so it's fit on training folds only, hyperparameters chosen by a stated procedure
(not by peeking at the test folds), and a table of *mean ± std* accuracy.
Then write `notes.md`: which model wins where and why, whether the differences are larger than the
fold-to-fold noise, and one situation each model would be your first choice.

## Can you explain it?

- Why does scaling change k-NN but not a tree?
- Why is a deep tree a *high-variance* model?
- What is wrong with picking the best of 20 models by their score on the test set?
