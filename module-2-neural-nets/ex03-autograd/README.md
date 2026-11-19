# 2.3 Autograd from scratch

**Goal:** see that backpropagation is nothing more than the chain rule applied
to a graph, by building a tiny automatic-differentiation engine (about 100 lines).
This is what PyTorch does for you with `loss.backward()`.

## Primer

Every arithmetic operation creates a **node** that remembers its inputs and knows its own
*local* derivative. Take `f = a·b + c`:

```
a ──┐
    ├─ (×) = d ──┐
b ──┘            ├─ (+) = f
c ────────────────┘
```

Walking backwards from `f`: `∂f/∂c = 1`, `∂f/∂d = 1`, and then by the chain rule
`∂f/∂a = ∂f/∂d · ∂d/∂a = 1 · b`. In code each node stores `grad` (the derivative of the
final output with respect to this node) and a small function that pushes its `grad` to its inputs.
Two details decide whether it works:

1. Visit nodes in **reverse topological order**, so a node's `grad` is complete before it is pushed on.
2. **Accumulate** with `+=`. A value used twice (like `a·a`) receives gradient from both uses.

Write your code in `autograd.py`. `pytest module-2-neural-nets/ex03-autograd` checks it.

## Steps

### 🟢 Step 1: a value that remembers
Create `Value(data)` with `.data`, `.grad = 0.0` and a record of which `Value`s it was
computed from. Implement `__add__` and `__mul__` for forward computation only (wrap plain
numbers in `Value`). Check `Value(2) * Value(-3) + Value(10)` has `.data == 4`.

### 🟢 Step 2: local derivatives
Give each op a `_backward` closure: for `+` it passes `out.grad` to both inputs, for `×` it passes
`out.grad * other.data` to each input. Remember `+=`. Call the closures by hand, in the right
order, for `f = a·b + c` and check `a.grad == −3`, `b.grad == 2`, `c.grad == 1`.

### 🟢 Step 3: let it walk by itself
Add `backward()`: build a topological ordering of the graph (depth-first search from the output),
set the output's `grad` to 1, then run each `_backward` in reverse order. Re-run step 2 with
one call.

### 🟢 Step 4: more operations
Add `__neg__`, `__sub__`, `__truediv__`, `__pow__` (constant exponent), `__radd__`, `__rmul__`,
`__rsub__`, `__rtruediv__`, and the methods `exp()`, `tanh()` and `relu()`. Derive each local
derivative on paper first.

### 🟢 Step 5: trust, but verify
Write a function of several variables mixing your ops, and compare each `.grad` to a numerical
derivative (central difference on plain floats). Then deliberately break `+=` into `=`
and watch `a * a` fail. Explain the failure in one sentence.

### 🔴 Step 6: build a network out of it
Using only `Value`, implement `Neuron(n_in)`, `Layer(n_in, n_out)` and `MLP(sizes)` with a `parameters()`
method and a `__call__`. `MLP([2, 4, 1])` must have 17 parameters. Train it on XOR with a loop of
forward, loss, `zero_grad`, `backward`, update. Report the loss curve.

### 🔴 Step 7: the cost of the graph
(a) Count the nodes in the graph for one forward pass of `MLP([2, 4, 1])` on one sample, and
extrapolate to a network of 1000×1000×1000 on a batch of 64. What does that tell you about why
real frameworks use arrays instead of scalar nodes?
(b) What happens if you forget to zero the gradients between steps? Show it.
(c) Explain why PyTorch's `optimizer.zero_grad()` exists, in terms of `+=`.

## Can you explain it?

- Why do we need a topological order, not just any order?
- Why is `+=` essential, and where does it matter in a real network?
- What is the difference between the forward pass and the backward pass in cost and memory?
