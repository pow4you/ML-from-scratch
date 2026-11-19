# 2.4 An MLP in numpy

**Goal:** build a real neural network with matrix operations, backward pass included,
and train it on XOR (which one neuron cannot learn) and on handwritten digits.

## Primer

A two-layer network on a batch `X` of shape `(N, d_in)`:

```
z1 = X  @ W1 + b1      (N, hidden)
a1 = tanh(z1)          (N, hidden)
z2 = a1 @ W2 + b2      (N, out)
p  = sigmoid(z2)       (N, out)
loss = mean BCE(p, y)
```

**Shapes are your compass.** The gradient of a matrix has the same shape as the matrix,
so `dW1` is `(d_in, hidden)`. Whenever a backward formula has the wrong shape, it's wrong.
Cache the intermediate values on the forward pass; the backward pass needs them.

Write your code in `mlp.py`. `helpers.py` has two tools you don't have to write:
`numeric_grad` (finite-difference gradient of any array) and `plot_decision_boundary`.
`pytest module-2-neural-nets/ex04-mlp-numpy` checks steps 1–5.

## Steps

### 🟢 Step 1: parameters and shapes
Implement `init_params(n_in, n_hidden, n_out, seed)` returning a dict with `W1, b1, W2, b2`.
Use normally distributed weights with a standard deviation around 0.5, and zero biases.
Print every shape. Two questions to keep in mind for exercise 2.5: why not all zeros? And what
happens if the scale is tiny (try 0.01 once your XOR works, and see what the loss does)?

### 🟢 Step 2: forward pass
Implement `forward(params, X)` returning `(p, cache)`. Run it on the four XOR points
and check `p` has shape `(4, 1)` and values strictly between 0 and 1. Implement `bce(p, y)`
(mean over all entries) and note the loss of an untrained net: it should be near `log 2 ≈ 0.69`.

### 🟢 Step 3: backward pass
On paper, work backwards through the primer's five lines. Implement
`backward(params, cache, y)` returning a dict of gradients with the same keys as `params`.

<details><summary>Hint</summary>
For sigmoid + BCE, the gradient at <code>z2</code> is <code>(p − y)</code> divided by the number of
entries in the mean. After that it is just: <code>dW2 = a1ᵀ @ dz2</code>, <code>db2</code> = sum over
rows, <code>da1 = dz2 @ W2ᵀ</code>, <code>dz1 = da1 · (1 − a1²)</code>, then the same pattern for layer 1.
</details>

### 🟢 Step 4: gradient check
Use `numeric_grad` to check every entry of every gradient on a random batch. The relative error should be below
`1e-6`. A gradient check is the single most useful debugging tool in this field: learn to trust it.
Run the tests.

### 🟢 Step 5: train XOR
Implement `train(params, X, y, lr, steps)` (full-batch gradient descent, update `params` in
place, return the list of losses). Train on XOR, plot the loss, plot the decision boundary,
and plot each hidden unit's activation over the input square. What is each hidden unit doing?

### 🔴 Step 6: any depth
Generalise: `MLP(sizes)` with `forward`, `backward` and `step(lr)` for any number of layers,
with a configurable hidden activation (tanh or relu). It must pass the same gradient check on a
network with three hidden layers. (No test file for this one: your gradient check is the test.)

### 🔴 Step 7: digits
Train your own network on `sklearn.datasets.load_digits` (8×8 images, 10 classes, scale pixels to
[0, 1]). You need a different output: **softmax + cross-entropy**. Derive its gradient (the answer is
beautifully similar to the sigmoid case), use mini-batches, and reach **at least 95% accuracy on a held-out test set**.
Report the confusion between digits: which pairs does your network mix up?

## Can you explain it?

- Why must you cache values on the forward pass?
- Why does a hidden layer let a network solve XOR?
- Why do we gradient-check against finite differences instead of just trusting the algebra?
