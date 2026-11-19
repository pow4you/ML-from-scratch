import numpy as np
from sklearn.linear_model import LinearRegression

from ols import fit_ols, predict


def make(n=200, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    true = np.array([1.5, -2.0, 0.5, 3.0][: k + 1])
    y = true[0] + X @ true[1:] + rng.normal(0, 0.01, n)
    return X, y, true


def test_recovers_known_coefficients():
    X, y, true = make()
    beta = fit_ols(X, y)
    assert beta.shape == (4,)
    np.testing.assert_allclose(beta, true, atol=0.01)


def test_matches_sklearn():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 4))
    y = rng.normal(size=100)
    beta = fit_ols(X, y)
    ref = LinearRegression().fit(X, y)
    np.testing.assert_allclose(beta[0], ref.intercept_, atol=1e-8)
    np.testing.assert_allclose(beta[1:], ref.coef_, atol=1e-8)


def test_predict_shape_and_values():
    X, y, true = make()
    beta = fit_ols(X, y)
    p = predict(X, beta)
    assert p.shape == (len(X),)
    np.testing.assert_allclose(p, y, atol=0.1)


def test_single_feature():
    x = np.arange(10, dtype=float).reshape(-1, 1)
    y = 4.0 + 2.0 * x[:, 0]
    np.testing.assert_allclose(fit_ols(x, y), [4.0, 2.0], atol=1e-9)
