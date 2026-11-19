# 2.5 Training parameters lab

**Goal:** develop a feel for the dials of training (learning rate, epochs, batch size,
capacity, regularisation, initialisation) by running controlled experiments instead of
memorising rules of thumb.

This exercise is experiment-driven, so there are no unit tests. Your evidence is plots and a
short findings file. Use the plumbing in `lab_utils.py` so the focus stays on the experiments.

## Primer

- **Learning rate:** the step size. The most important dial. Look at it on a log scale.
- **Epoch:** one pass over the training data. **Batch size:** how many samples per gradient step.
- **Capacity:** how many hidden units. More capacity means more ability to fit, including the noise.
- **Overfitting:** training score keeps improving while validation score stalls or drops.
- **Regularisation (L2, `alpha`):** a penalty on large weights that trades fit for smoothness.
- **Three datasets:** *train* fits the weights, *validation* picks the settings, *test* is touched once, at the very end.

Rule for the whole exercise: **change one thing at a time, and repeat with at least 3 seeds before believing a difference.**

## Steps

### 🟢 Step 1: the baseline
Use `lab_utils.run(...)` with its defaults (plain SGD, 32 hidden units, `lr=0.1`). Record train
and validation accuracy and plot the loss curve (`loss_curve_`). This is your reference run.

### 🟢 Step 2: learning-rate sweep
Run `lr` over `1e-4, 1e-3, 1e-2, 1e-1, 1, 10`. Plot final loss and validation accuracy against `lr` on a log
x-axis. Name the three regimes you see (too small, good, too big) and describe what each loss curve looks like.

### 🟢 Step 3: when does more training hurt?
Use only **150 training samples** (`load_split(n_train=150)`) with 64 hidden units and
`learning_rate_init=1.0` so overfitting is visible. Train one epoch at a time with `partial_fit` for
300 epochs, recording training and validation loss (`sklearn.metrics.log_loss` on `predict_proba`) and accuracy
each epoch. Plot both. Mark the epoch where the validation loss stops improving. Loss shows overfitting more
clearly than accuracy does: why? What would early stopping buy you?

### 🟢 Step 4: capacity and regularisation
With the 150-sample set, train networks with 2, 8, 32 and 128 hidden units and with `alpha` in
`0, 1e-3, 1e-1, 1`. Show the validation accuracy as a heat map (hidden units × alpha). Where is the sweet spot?
Does the best `alpha` depend on the capacity?

### 🔴 Step 5: batch size and initialisation
(a) Compare batch sizes 1, 16, 64 and full-batch at a **fixed number of epochs**. Plot
validation accuracy against epochs *and* against wall-clock time. Which batch size is "best" depends on the
x-axis. Explain why.
(b) Using your numpy network from 2.4 on XOR: initialise all weights to zero and train. Look at the
hidden units after training: what do you see, and why can gradient descent never fix it? Now try
a tiny init (std 0.01) and a large one (std 3). Describe the three failures.

### 🔴 Step 6: a fair little tuning study
Tune `lr`, `hidden_layer_sizes` and `alpha` by **random search** (20 configurations, drawn
from ranges you choose, `lr` log-uniform). Select the best on **validation**, then evaluate that one configuration
on the **test set exactly once**. Repeat the whole study with 3 different seeds. Write `FINDINGS.md` with: your
ranges, a table of the 3 winning configs and their validation vs test scores, and a paragraph on what the
seed-to-seed variation says about how much to trust small differences.

## Can you explain it?

- Why is the learning rate searched on a log scale?
- How would you tell, from curves alone, that a network is overfitting? Underfitting?
- Why must the test set be used only once?
- Why does zero initialisation break learning even though gradients are non-zero?
