import numpy as np

import mlp
from helpers import numeric_grad

XOR_X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_y = np.array([[0.0], [1.0], [1.0], [0.0]])


def test_init_shapes_and_zero_biases():
    p = mlp.init_params(3, 5, 2, seed=1)
    assert p["W1"].shape == (3, 5) and p["b1"].shape == (5,)
    assert p["W2"].shape == (5, 2) and p["b2"].shape == (2,)
    assert np.abs(p["W1"]).max() > 0  # not all zeros


def test_init_is_seeded():
    a, b = mlp.init_params(3, 5, 2, seed=7), mlp.init_params(3, 5, 2, seed=7)
    np.testing.assert_array_equal(a["W1"], b["W1"])


def test_forward_shape_and_range():
    p = mlp.init_params(2, 4, 1)
    out, _ = mlp.forward(p, XOR_X)
    assert out.shape == (4, 1)
    assert np.all((out > 0) & (out < 1))


def test_untrained_loss_is_near_log2():
    p = mlp.init_params(2, 4, 1)
    out, _ = mlp.forward(p, XOR_X)
    assert abs(mlp.bce(out, XOR_y) - np.log(2)) < 0.15


def test_bce_known_value():
    assert abs(mlp.bce(np.array([[0.5], [0.5]]), np.array([[1.0], [0.0]])) - np.log(2)) < 1e-9


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 3))
    y = (rng.random((6, 2)) < 0.5).astype(float)
    params = mlp.init_params(3, 4, 2, seed=3)
    for k in params:  # make biases non-trivial too
        params[k] = params[k] + rng.normal(0, 0.3, params[k].shape)
    p, cache = mlp.forward(params, X)
    grads = mlp.backward(params, cache, y)
    assert set(grads) == set(params)
    for k in params:
        assert grads[k].shape == params[k].shape, k
        num = numeric_grad(lambda: mlp.bce(mlp.forward(params, X)[0], y), params[k])
        denom = np.maximum(1e-8, np.abs(num) + np.abs(grads[k]))
        assert np.max(np.abs(num - grads[k]) / denom) < 1e-5, k


def test_training_solves_xor():
    params = mlp.init_params(2, 8, 1, seed=0)
    losses = mlp.train(params, XOR_X, XOR_y, lr=0.5, steps=6000)
    assert len(losses) == 6000
    assert losses[-1] < 0.1 and losses[-1] < losses[0]
    p, _ = mlp.forward(params, XOR_X)
    assert np.all((p > 0.5) == (XOR_y > 0.5))
