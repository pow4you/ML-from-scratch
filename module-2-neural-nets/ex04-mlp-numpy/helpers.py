"""Plumbing for exercise 2.4: you don't need to write these."""

import matplotlib.pyplot as plt
import numpy as np


def numeric_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of the scalar function f(x) with respect to the array x.

    `x` is modified temporarily and restored, so f may close over it.
    """
    g = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        hi = f()
        x[i] = old - eps
        lo = f()
        x[i] = old
        g[i] = (hi - lo) / (2 * eps)
    return g


def plot_decision_boundary(predict_fn, X, y, ax=None, lim=(-0.5, 1.5)):
    """predict_fn maps an (M, 2) array to probabilities of shape (M,) or (M, 1)."""
    ax = ax or plt.gca()
    xs = np.linspace(*lim, 200)
    gx, gy = np.meshgrid(xs, xs)
    grid = np.c_[gx.ravel(), gy.ravel()]
    z = np.asarray(predict_fn(grid)).reshape(gx.shape)
    ax.contourf(gx, gy, z, levels=20, cmap="RdBu", alpha=0.6)
    ax.contour(gx, gy, z, levels=[0.5], colors="k")
    ax.scatter(X[:, 0], X[:, 1], c=np.ravel(y), cmap="RdBu", edgecolors="k", s=80)
    return ax
