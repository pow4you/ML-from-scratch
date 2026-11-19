# 3.4 Images and convolutions

**Goal:** understand *why* convolutional networks exist (not just that they win), by implementing a
convolution yourself and then comparing a small CNN with the MLP you already know.

Needs PyTorch (`pip install -r requirements-torch.txt`).

## Primer

A fully connected layer on an image treats every pixel independently, with its own weights, and ignores
the fact that neighbouring pixels belong together. A **convolution** slides one small kernel over the
whole image: the same weights everywhere (*weight sharing*), looking only at a small neighbourhood (*locality*).
A kernel is a pattern detector; the output (a *feature map*) shows where that pattern occurs. **Pooling**
shrinks feature maps and makes detection tolerant to small shifts.

What deep-learning libraries call "convolution" is technically *cross-correlation* (the kernel is not flipped).
Do the same.

Write your code in `conv.py` and `cnn.py`. `pytest module-3-applied-ml/ex04-images-and-convolutions` checks them.

## Steps

### 🟢 Step 1: a convolution by hand
Implement `conv2d(img, kernel)` ("valid" mode, stride 1, no padding) with plain loops first. Apply the Sobel
kernels `[[-1,0,1],[-2,0,2],[-1,0,1]]` and its transpose to a digit from `load_digits` and plot the input and
both outputs. What does each kernel respond to? What is the output shape for an 8×8 image and a 3×3 kernel?

### 🟢 Step 2: pooling and the parameter bill
Implement `max_pool2d(img, size)` (non-overlapping windows, stride = size, drop any remainder). Then do some
arithmetic on paper, in `notes.md`: for a 28×28 greyscale image, how many weights does
(a) a fully connected layer to 100 units have? (b) a conv layer with 100 filters of size 3×3? Why is the
difference not just "smaller", but a different *kind* of model?

### 🟢 Step 3: a small CNN
Implement `DigitsCNN` in `cnn.py`: `Conv2d(1→8, 3)` → ReLU → `MaxPool2d(2)` → flatten → `Linear(→10)`, taking
input of shape `(N, 1, 8, 8)` and returning logits. Work out the tensor shape after each layer by hand, then confirm by
printing them. Train it with your loop from 3.2 and compare with your MLP at a *similar number of parameters*.

### 🔴 Step 4: does augmentation help?
Cut the training set to 200 samples. Implement on-the-fly augmentation (random shift of ±1 pixel, small random
rotation using `scipy.ndimage`) and measure test accuracy for both the CNN and the MLP, with and without it, over 5 seeds.
Does it help? Does it help one model more? Be ready for a surprising answer: these images are only 8×8 pixels,
so test each transformation alone (shift only, rotation only) and at several strengths before you conclude anything,
and look at what an augmented digit actually looks like. Explain why *horizontal flips* would be a bad augmentation
for digits, and name one dataset where they'd be fine.

### 🔴 Step 5: look inside
Plot the 8 learned first-layer filters. Show 12 misclassified test digits with true and predicted labels, and write a
paragraph on what the network confuses and why. For one misclassified digit, compute a **saliency map**
(the gradient of the predicted logit with respect to the input pixels) and plot it. Does it highlight what you'd expect?

## Can you explain it?

- What do weight sharing and locality buy you, in terms of parameters and in terms of what the model can generalise to?
- Why does pooling make a model tolerant to small shifts?
- When would a plain MLP still be the right choice?
