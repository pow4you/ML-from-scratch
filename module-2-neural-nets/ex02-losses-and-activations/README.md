# 2.2 Losses and activations

**Goal:** go from predicting numbers to predicting probabilities, and understand why
that needs a different loss, and why the choice of activation function decides whether
a deep network can learn at all.

## Primer

A **logistic neuron** squashes a linear score into a probability:

```
z = w·x + b          p = σ(z) = 1 / (1 + e⁻ᶻ)
```

For binary labels `y ∈ {0, 1}` the natural loss is **binary cross-entropy**:

```
BCE = −mean( y·log(p) + (1−y)·log(1−p) )
```

It is the negative log-likelihood of the labels under the model: it punishes confident
wrong answers far more than squared error does. A neural net is a stack of such
neurons, with a non-linear **activation** between layers. Without non-linearities,
a stack of linear layers is just one linear layer.

Write your code in `logistic.py`. `pytest module-2-neural-nets/ex02-losses-and-activations` checks it.

## Steps

### 🟢 Step 1: the sigmoid
Implement `sigmoid` so it does not overflow for `z = −1000` or `z = +1000`, and
`sigmoid_grad(z)`. Plot both on `[−10, 10]`. Where is the derivative largest, and what is
its maximum? What is it at `|z| = 10`?

<details><summary>Hint</summary>
For large negative z, <code>exp(-z)</code> overflows. Use a different but equivalent formula
for negative inputs, e.g. with <code>np.where</code> (careful: it computes both branches).
</details>

### 🟢 Step 2: a logistic neuron, forward only
Make two 2-D blobs with `sklearn.datasets.make_blobs(centers=2, random_state=0)`.
Implement `forward(w, b, X)` returning probabilities. Try random `w`. Plot the points
colored by label and draw the **decision boundary** `w·x + b = 0`. Compute accuracy at threshold 0.5.

### 🟢 Step 3: cross-entropy vs squared error
Implement `bce(p, y)` (clip `p` away from 0 and 1). For a true label `y = 1`, tabulate
both losses for `p = 0.99, 0.5, 0.1, 0.01, 0.0001`. How does each loss grow as the model
becomes confidently wrong? Which one would give a stronger learning signal for the worst predictions?

### 🔴 Step 4: the gradient, and training
Derive `∂BCE/∂z` for a sigmoid output. It simplifies a lot. Use it to implement
`grads(w, b, X, y)` (mean over samples; `dw` has shape `(d,)`), check by finite
differences, and write `train(X, y, lr, steps)`. Plot the decision boundary at steps 0, 10, 100
and 1000.

### 🔴 Step 5: the activation zoo
Implement `relu`, `relu_grad`, `leaky_relu`, `leaky_relu_grad` and `tanh_grad`.
Then run an experiment: push a gradient backwards through a chain of 10 scalar neurons,
with weight 1 and a pre-activation of 0.5 at each, for sigmoid, tanh and ReLU. Tabulate
the gradient reaching the first layer. Explain **vanishing gradients** with your
table, explain the **dying ReLU** problem, and say when you'd choose each activation.

## Can you explain it?

- Why can't a network of only linear layers learn XOR?
- Why is cross-entropy a better fit than squared error for probabilities?
- Why did ReLU replace sigmoid in the hidden layers of deep networks?
