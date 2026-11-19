# 3.2 PyTorch

**Goal:** move to a real framework *knowing what it does for you*. Because you built autograd (2.3)
and a numpy MLP (2.4), every PyTorch call here should look like something you wrote yourself.

Install the extra requirements first: `pip install -r requirements-torch.txt`.

## Primer

PyTorch gives you: **tensors** (numpy arrays that remember their history), **autograd**
(`loss.backward()` fills `.grad` on every parameter, accumulating with `+=`),
**modules** (`nn.Module` bundles parameters and a `forward`), **optimizers** (update parameters from `.grad`),
and **data loaders** (batching and shuffling).

The canonical training step is five lines, and you know why each one exists:

```python
optimizer.zero_grad()      # because gradients accumulate with +=
loss = loss_fn(model(xb), yb)   # forward
loss.backward()            # backward
optimizer.step()           # update
```

Write your code in `digits_net.py`. `pytest module-3-applied-ml/ex02-pytorch` checks it.

## Steps

### 🟢 Step 1: tensors and autograd
Recompute `f = a·b + c` from exercise 2.3 with `torch.tensor(..., requires_grad=True)` and call `.backward()`.
Check the gradients match your `Value` engine. Call `backward()` twice without zeroing `.grad` and watch
the accumulation. Use `torch.no_grad()` to evaluate and confirm no graph is built.

### 🟢 Step 2: data and model
Load `sklearn.datasets.load_digits` (scale pixels to [0, 1]) and split into train and test. Implement `DigitsNet(hidden=32)`
(64 → hidden → ReLU → 10 logits) as an `nn.Module`. Print the parameter count and check
by hand that it's `64·32 + 32 + 32·10 + 10`. Check the output shape is `(N, 10)` for a batch of `N`.

### 🟢 Step 3: the training loop
Implement `train(model, X, y, epochs, lr, batch_size=32, seed=0)`: use `DataLoader` with shuffling,
`nn.CrossEntropyLoss` (it expects *logits*, so why is there no softmax in the model?), and
**`torch.optim.SGD`** with the given learning rate. Return the mean loss of each epoch. Implement `accuracy(model, X, y)` taking numpy arrays.
Reach at least 95% test accuracy. Then (a) comment out `zero_grad` and watch the training break, (b) swap to `Adam` and compare.

### 🔴 Step 4: reproduce your numpy network exactly
Build a PyTorch network equivalent to your `mlp.py` from exercise 2.4 (tanh hidden layer, sigmoid output, mean BCE; use
`BCEWithLogitsLoss`), in **float64**. Load your numpy `init_params` weights into it (careful: `nn.Linear`
stores its weight transposed). On the XOR data, compare the loss and **every gradient** of your hand-written `backward` against
autograd's. They must agree to better than `1e-8`. If they don't, one of them has a bug: find out which.

### 🔴 Step 5: the regularisation toolkit
With only 300 training samples, add and ablate: dropout, weight decay, a learning-rate schedule
(`StepLR` or cosine), and early stopping on a validation split. Report a table of test accuracy for each
combination that matters, with mean ± std over 5 seeds. Which tools helped, which were noise, and how do you know?

## Can you explain it?

- Why does PyTorch need `optimizer.zero_grad()`?
- Why does `CrossEntropyLoss` take logits rather than probabilities?
- What does `model.eval()` change, and what does `torch.no_grad()` change? (They are different.)
