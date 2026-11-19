import numpy as np

from neuron import gradient_descent, grads, mse, predict


def data(n=50, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = 3 * x + 2 + rng.normal(0, 0.5, n)
    return x, y


def test_predict_and_mse_known_values():
    x = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(predict(2.0, 1.0, x), [1.0, 3.0, 5.0])
    y = np.array([1.0, 4.0, 5.0])  # errors: 0, -1, 0
    assert abs(mse(2.0, 1.0, x, y) - 1 / 3) < 1e-12


def test_grads_match_finite_differences():
    x, y = data()
    for w, b in [(0.0, 0.0), (3.0, 2.0), (-1.5, 7.0), (5.0, -1.0)]:
        dw, db = grads(w, b, x, y)
        eps = 1e-6
        num_dw = (mse(w + eps, b, x, y) - mse(w - eps, b, x, y)) / (2 * eps)
        num_db = (mse(w, b + eps, x, y) - mse(w, b - eps, x, y)) / (2 * eps)
        assert abs(dw - num_dw) < 1e-5 * max(1, abs(num_dw))
        assert abs(db - num_db) < 1e-5 * max(1, abs(num_db))


def test_gradient_is_zero_at_the_optimum():
    x = np.linspace(-1, 1, 20)
    y = 3 * x + 2
    dw, db = grads(3.0, 2.0, x, y)
    assert abs(dw) < 1e-12 and abs(db) < 1e-12


def test_gradient_descent_recovers_the_line():
    x, y = data()
    w, b, losses = gradient_descent(x, y, lr=0.05, steps=500, w0=-1.5, b0=7.0)
    assert abs(w - 3) < 0.15 and abs(b - 2) < 0.2
    assert len(losses) == 500
    assert losses[-1] < losses[0] / 10
    assert all(np.diff(losses) <= 1e-12)  # monotone for a small learning rate


def test_too_big_a_learning_rate_diverges():
    x, y = data()
    _, _, losses = gradient_descent(x, y, lr=1.0, steps=50, w0=0.0, b0=0.0)
    assert losses[-1] > losses[0]
