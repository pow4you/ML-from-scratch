import warnings

import numpy as np
from sklearn.datasets import make_blobs

import logistic as lg


def test_sigmoid_values_and_stability():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overflow warning fails the test
        s = lg.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(s, [0.0, 0.5, 1.0], atol=1e-12)
    assert np.all(np.isfinite(s))


def test_sigmoid_grad():
    z = np.linspace(-8, 8, 33)
    g = lg.sigmoid_grad(z)
    assert abs(g.max() - 0.25) < 1e-12
    assert abs(lg.sigmoid_grad(np.array([0.0]))[0] - 0.25) < 1e-12
    assert lg.sigmoid_grad(np.array([10.0]))[0] < 1e-4


def test_bce_known_values():
    y = np.array([1.0, 0.0])
    p = np.array([0.5, 0.5])
    assert abs(lg.bce(p, y) - np.log(2)) < 1e-9
    assert lg.bce(np.array([0.99, 0.01]), y) < 0.02
    assert np.isfinite(lg.bce(np.array([0.0, 1.0]), y))  # clipping works


def test_forward_shape_and_range():
    X = np.random.default_rng(0).normal(size=(7, 3))
    p = lg.forward(np.array([1.0, -1.0, 0.5]), 0.1, X)
    assert p.shape == (7,)
    assert np.all((p > 0) & (p < 1))


def test_grads_match_finite_differences():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = (rng.random(30) < 0.5).astype(float)
    w, b = rng.normal(size=3), 0.3
    dw, db = lg.grads(w, b, X, y)
    assert dw.shape == (3,)
    f = lambda w_, b_: lg.bce(lg.forward(w_, b_, X), y)
    eps = 1e-6
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        num = (f(w + e, b) - f(w - e, b)) / (2 * eps)
        assert abs(dw[i] - num) < 1e-6
    assert abs(db - (f(w, b + eps) - f(w, b - eps)) / (2 * eps)) < 1e-6


def test_training_separates_blobs():
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=0)
    w, b, losses = lg.train(X, y, lr=0.1, steps=500)
    assert len(losses) == 500
    assert losses[-1] < losses[0]
    acc = np.mean((lg.forward(w, b, X) > 0.5) == y)
    assert acc > 0.93


def test_activations_and_their_gradients():
    z = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_allclose(lg.relu(z), [0, 0, 3])
    np.testing.assert_allclose(lg.relu_grad(np.array([-2.0, 3.0])), [0, 1])
    np.testing.assert_allclose(lg.leaky_relu(z, 0.1), [-0.2, 0, 3])
    np.testing.assert_allclose(lg.leaky_relu_grad(np.array([-2.0, 3.0]), 0.1), [0.1, 1])
    np.testing.assert_allclose(lg.tanh_grad(np.array([0.0])), [1.0])
    np.testing.assert_allclose(lg.tanh_grad(np.array([1.0])), [1 - np.tanh(1.0) ** 2])
