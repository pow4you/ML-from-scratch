# 2.1 One neuron learns a line

**Goal:** understand gradient descent completely on the smallest model there is:
two numbers, `w` and `b`. Everything in deep learning is this, scaled up.

## Primer

The model is `ŷ = w·x + b`. The loss measures how wrong it is on the data:

```
L(w, b) = mean( (w·x + b − y)² )
```

Treat `L` as a landscape over the plane of `(w, b)`. For this model it is a smooth bowl.
The **gradient** `(∂L/∂w, ∂L/∂b)` points uphill, so we step the other way:

```
w ← w − lr · ∂L/∂w        b ← b − lr · ∂L/∂b
```

`lr` (the learning rate) is the step size. Too small and you crawl; too big and you
overshoot the bottom, and can fly off to infinity.

Write your code in `neuron.py`. `pytest module-2-neural-nets/ex01-one-neuron` checks the pieces.

## Steps

### 🟢 Step 1: data and eyes
In `starter.py`, generate `y = 3x + 2 + noise` for 50 points and plot them. Before computing
anything, guess `w` and `b` by eye.

### 🟢 Step 2: the loss and its bowl
Implement `predict` and `mse`. Evaluate the loss at `(0,0)`, `(3,2)` and `(5,−1)`.
Then compute the loss on a grid of `(w, b)` and draw a contour plot. Mark the point
`(3,2)`. The contours are ellipses: along which axis is the bowl steeper, and why?

### 🟢 Step 3: the gradient, by hand
On paper, differentiate `L` with respect to `w` and `b` (chain rule: `L = mean(e²)` with
`e = w·x + b − y`). Implement `grads(w, b, x, y)`. Check it against the **finite difference**
`(L(w+ε) − L(w−ε)) / 2ε` for both parameters. Run the tests.

<details><summary>Hint</summary>
The derivative of e² is 2e, and e depends on w through x. For b, the "x" is just 1.
</details>

### 🟢 Step 4: walk downhill
Implement `gradient_descent(x, y, lr, steps, w0, b0)` returning `(w, b, losses)`,
where `losses` has one entry per step (the loss *before* that update).
Start at `(−1.5, 7)` with `lr = 0.05` and 100 steps. Plot the loss curve on a log
y-axis, then draw the path of `(w, b)` on top of your contour plot.

### 🔴 Step 5: the speed limit
Find, **by experiment**, the largest learning rate that still converges on your data.
Then predict it **from theory**: the Hessian of `L` is `H = 2·[[mean(x²), mean(x)], [mean(x), 1]]`,
and gradient descent on a quadratic converges iff `lr < 2/λ_max(H)`. Check that your
experiment agrees with the theory to within 5%.
Then explain, using that formula, why a very elongated bowl makes learning slow.

### 🔴 Step 6: noise and scale
(a) Implement mini-batch SGD with batch size B (reshuffle each epoch). Compare loss
curves for B = 1, 8, 50. What does the noise do near the bottom? Is it ever useful?
(b) Replace `x` with `x·100 + 500` (think "square metres" instead of standardised
numbers). Run the lr that worked before. What happens? Fix it by standardising `x`, and
explain *why* this works, using the formula from step 5.

## Can you explain it?

- Why do we subtract the gradient rather than add it?
- What does the learning rate trade off?
- Why does standardising inputs make training faster?
